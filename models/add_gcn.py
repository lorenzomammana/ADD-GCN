import torch
import torch.nn as nn
from models.mobilevit import MobileViT


class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):
        super(DynamicGraphConvolution, self).__init__()

        self.static_adj = nn.Sequential(nn.Conv1d(num_nodes, num_nodes, 1, bias=False), nn.LeakyReLU(0.2))
        self.static_weight = nn.Sequential(nn.Conv1d(in_features, out_features, 1), nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features * 2, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

    def forward_static_gcn(self, x):
        x = self.static_adj(x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x

    def forward_construct_dynamic_graph(self, x):
        ### Model global representations ###
        x_glb = self.gap(x)
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))

        ### Construct the dynamic correlation matrix ###
        x = torch.cat((x_glb, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)
        dynamic_adj = torch.sigmoid(dynamic_adj)
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        """D-GCN module

        Shape:
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        out_static = self.forward_static_gcn(x)
        x = x + out_static  # residual
        dynamic_adj = self.forward_construct_dynamic_graph(x)
        x = self.forward_dynamic_gcn(x, dynamic_adj)
        return x, dynamic_adj


class ADD_GCN(nn.Module):
    def __init__(self, model, num_classes, skip_gcn: bool = False):
        super(ADD_GCN, self).__init__()

        if isinstance(model, MobileViT):
            self.features = model
        else:
            self.features = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            )

        self.num_classes = num_classes
        self.skip_gcn = skip_gcn

        if not self.skip_gcn:
            self.fc = nn.Conv2d(model.fc.in_features, num_classes, (1, 1), bias=False)

            self.conv_transform = nn.Conv2d(model.fc.in_features, model.fc.in_features // 2, (1, 1))
            self.relu = nn.LeakyReLU(0.2)

            self.gcn = DynamicGraphConvolution(model.fc.in_features // 2, model.fc.in_features // 2, num_classes)

            self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
            self.last_linear = nn.Conv1d(model.fc.in_features // 2, self.num_classes, 1)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            self.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, model.fc.in_features // 2),
                nn.Dropout(p=0.3),
                nn.Linear(model.fc.in_features // 2, model.fc.in_features // 8),
                nn.Dropout(p=0.3),
                nn.Linear(model.fc.in_features // 8, self.num_classes),
            )

    def forward_feature(self, x):
        x = self.features(x)
        return x

    def forward_classification_sm(self, x):
        """Get another confident scores {s_m}.

        Shape:
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out) # C_out: num_classes
        """
        x = self.fc(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.topk(1, dim=-1)[0].mean(dim=-1)
        return x

    def forward_sam(self, x):
        """SAM module

        Shape:
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        mask = self.fc(x)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)

        x = self.conv_transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask)

        return x

    def forward_dgcn(self, x):
        x, adj_mat = self.gcn(x)
        return x, adj_mat

    def forward(self, x):
        x = self.forward_feature(x)

        if not self.skip_gcn:
            out1 = self.forward_classification_sm(x)

            v = self.forward_sam(x)  # B*1024*num_classes
            z, adj_mat = self.forward_dgcn(v)
            z = v + z

            out2 = self.last_linear(z)  # B*1*num_classes
            mask_mat = self.mask_mat.detach()
            out2 = (out2 * mask_mat).sum(-1)

            return out1, out2, adj_mat
        else:
            x = self.avgpool(x)
            out1 = self.fc(x.view(x.shape[0], -1))
            return out1, out1, None

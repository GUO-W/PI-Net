##
## Software PI-Net: Pose Interacting Network for Multi-Person Monocular 3D Pose Estimation
## Copyright Inria and Polytechnic University of Catalonia  [to be checked] (do the other people you collaborate come from this university ?)
## Year 2021
## Contact : wen.guo@inria.fr
##
## The software PI-Net is provided under MIT License.
##

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import cfg

def get_model(pretrained = False):
    model = ContextualRescorer()
    return model


class ContextualRescorer(nn.Module):
    def __init__(self):
        super(ContextualRescorer, self).__init__()

        self.hidden_size = 256
        self.input_size = 17*3
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_layers = 3
        self.directions = 2
        self.attention_type =  "general"
        self.loss_type = 'mse'

        self.h0 = nn.Parameter(
            torch.zeros(self.num_layers * self.directions, 1, self.hidden_size),
            requires_grad=True,
        )

        self.rnn = nn.GRU(
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional= True,
            batch_first=True,
            dropout= 0
        )

        if self.attention_type != "none":
            layer_size = self.hidden_size * self.directions*2
        else:
            layer_size = self.hidden_size * self.directions
        self.linear1 = nn.Linear(layer_size, 256) #conv1d
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 17*3, bias=False)
        self.b1 = nn.LayerNorm(512*2, elementwise_affine=True)
        self.b2 = nn.LayerNorm(256, elementwise_affine=True)
        self.b3 = nn.LayerNorm(128, elementwise_affine=True)
        self.relu = nn.ReLU()
        self.Wa = nn.Linear(self.hidden_size * self.directions, self.hidden_size * self.directions, bias=True)

    def init_hidden(self, batch_size=1):
        h0 = self.h0.repeat(1, batch_size, 1)
        return h0

    def forward(self, input_, lengths, target=None): #lengths = tensor of nb_instances/img in one batch  #, lengths, mask):
        n = lengths.max() # n = patched nb of person in one img / patch

        eps = torch.ones(1) * 1e-4
        eps = eps.cuda()
        input_mean = input_.mean(dim=1, keepdim=True)
        input_std = input_.std(dim=1, keepdim=True)
        input_ = (input_ - input_mean) / (input_std + eps)

        # b * (17*n) * 3 -> b * n * 51
        input_ = torch.reshape(input_, (-1, n, 51))
        batch_size, _, _ = input_.size()

        ### rnn
        h0 = self.init_hidden(batch_size=batch_size)
        packed_input = pack_padded_sequence(
            input_, lengths, batch_first=True, enforce_sorted=False
        )
        hidden, _ = self.rnn(packed_input, h0)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)  # unpack sequence

        ### classifier
        if self.attention_type != "none":
            context,alpha = self.attention(hidden, mask=None)
            output = self.classifier(hidden, context, input_)
        else:
            output = self.classifier(hidden, "none", input_)

        # b * n * 51 -> b * (17*n) * 3
        output = torch.reshape(output, (-1, 17*n, 3))

        if target is None:
            output = output * (input_std + eps) + input_mean
            if cfg.vis_A:
                return output,alpha
            else:
                return output
        else:
            coord_cam = output

            target_cam = target['coord_cam']
            target_cam = (target_cam - target_cam.mean(dim=1, keepdim=True)) / (target_cam.std(dim=1, keepdim=True) + eps)
            target_vis = target['vis']
            target_have_depth = target['have_depth']

            loss_cam = torch.abs(coord_cam - target_cam) * target_vis
            loss_coord = loss_cam

            loss_x = loss_coord[:,:,0]
            loss_y = loss_coord[:,:,1]
            loss_z = loss_coord[:,:,2] * target_have_depth
            loss_coord = (loss_x + loss_y + loss_z * target_have_depth)/3.
            return loss_coord, loss_x, loss_y, loss_z


    def classifier(self, hidden, context, input_):
        hidden = torch.cat((hidden, context), dim=2)

        hidden = hidden.reshape(-1,512*2)
        hidden = self.b1(hidden)
        hidden = self.relu(self.linear1(hidden))
        hidden = self.b2(hidden)
        hidden = self.linear2(hidden)
        hidden = self.b3(hidden)
        hidden = self.linear3(hidden)
        return hidden

    def attention(self, hidden, mask=None):
        B, L, H = hidden.size()
        hs = self.Wa(hidden)
        scores = torch.bmm(hidden, hs.transpose(1, 2))
        alpha = torch.softmax(scores, dim=2)
        context = torch.bmm(alpha, hidden)
        return context,alpha

if __name__ == '__main__':
    '''
    from torchsummary import summary
    network = ContextualRescorer().cuda()
    summary(network,(4,17*5,3))
    '''
    network = ContextualRescorer().cuda()
    #l = torch.tensor([5,5,4,4]).cuda()
    #a = torch.rand(4,17*5,3).cuda()
    #output = network.forward(a, l, target=None)
    #print(network)#output.size())

    params = list(network.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l*=j
        k = k+l
    print("all params:"+ str(k/1000000))

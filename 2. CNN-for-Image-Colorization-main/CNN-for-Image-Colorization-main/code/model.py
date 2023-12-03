from torch import batch_norm, max_pool2d
import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class DownConv(nn.Module):
    def __init__(
            self, kernel, in_channels, out_channels, NF
    ):
        super(DownConv, self).__init__()

        # properties of class
        self.downconv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels, kernel_size=kernel, stride=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(NF),
            nn.ReLU()
        )

    def forward(self, x):
        ##############################################################################################
        #                                       Your Code                                            #
        x = self.downconv(x)
        ##############################################################################################
        return x


class UpConv(nn.Module):
    def __init__(
            self, kernel, in_channels, out_channels, NF
    ):
        super(UpConv, self).__init__()

        # properties of class
        self.upconv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels, kernel_size=kernel, stride=1),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(NF),
            nn.ReLU()
        )

    def forward(self, x):
        ##############################################################################################
        #                                       Your Code                                            #
        x = self.upconv(x)
        ##############################################################################################
        return x


class Bottleneck(nn.Module):
    def __init__(
            self, kernel, in_channels, out_channels, NF
    ):
        super(Bottleneck, self).__init__()

        # properties of class
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels, kernel_size=kernel, stride=1),
            nn.BatchNorm2d(NF),
            nn.ReLU()
        )

    def forward(self, x):
        ##############################################################################################
        #                                       Your Code                                            #
        x = self.bottleneck(x)
        ##############################################################################################
        return x


class BaseModel(nn.Module):
    def __init__(
            self, kernel, num_filters, num_colors, in_channels=1, padding=1
    ):
        super(BaseModel, self).__init__()

  #       implementation using DownConv, UpConv and Bottlnek classes       #

    #     self.ups = nn.ModuleList()
    #     self.downs = nn.ModuleList()
    #     # Other properties if needed
    #     # Down part of the model, bottleneck, Up part of the model, final conv
    #     ##############################################################################################
    #     #                                       Your Code                                            #
    #     self.downs.append(DownConv(
    #         kernel=kernel, in_channels=in_channels, out_channels=num_filters, NF=num_filters))
    #     self.downs.append(DownConv(
    #         kernel=kernel, in_channels=num_filters, out_channels=2*num_filters, NF=2*num_filters))
    #     self.downs.append(Bottleneck(
    #         kernel=kernel, in_channels=2*num_filters, out_channels=2*num_filters, NF=2*num_filters))

    #     self.ups.append(
    #         UpConv(kernel=kernel, in_channels=2*num_filters, out_channels=num_filters, NF=num_filters))
    #     self.ups.append(
    #         UpConv(kernel=kernel, in_channels=num_filters, out_channels=num_colors, NF=num_colors))
    #     self.ups.append(nn.Conv2d(in_channels=num_colors,
    #                                   out_channels=num_colors, kernel_size=kernel, stride=1))
    #     ##############################################################################################

    # def forward(self, x):
    #     ##############################################################################################
    #     #                                       Your Code                                            #
    #     first = self.downs[0](x)
    #     second = self.downs[1](first)
    #     third = self.downs[2](second)
    #     fourth = self.downs[1](third)
    #     fifth = self.downs[1](fourth)
    #     sixth = self.downs[1](fifth)
    #     ##############################################################################################
    #     return sixth


#       implementation using Sequential layers inside the BaseModel class

        padding = kernel // 2

        self.downconv1 = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel, padding=padding),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        self.downconv2 = nn.Sequential(
            nn.Conv2d(num_filters, 2*num_filters, kernel, padding=padding),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(2*num_filters),
            nn.ReLU()
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(2*num_filters, 2*num_filters, kernel, padding=padding),
            nn.BatchNorm2d(2*num_filters),
            nn.ReLU()
        )

        self.upconv1 = nn.Sequential(
            nn.Conv2d(2*num_filters, num_filters, kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        self.upconv2 = nn.Sequential(
            nn.Conv2d(num_filters, num_colors, kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_colors),
            nn.ReLU()
        )

        self.last_conv = nn.Conv2d(
            num_colors, num_colors, kernel, padding=padding)

    def forward(self, x):
        first = self.downconv1(x)
        second = self.downconv2(first)
        third = self.bottleneck(second)
        fourth = self.upconv1(third)
        fifth = self.upconv2(fourth)
        output = self.last_conv(fifth)
        return output


class MyConv2d(nn.Module):  # required for CustomUNET class
    def __init__(self, in_channels, out_channels, kernel_size, padding=None):
        super(MyConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if padding is None:
            self.padding = kernel_size // 2
        else:
            self.padding = padding
        self.weight = nn.parameter.Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.parameter.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, padding=self.padding)


class CustomUNET(nn.Module):
    def __init__(self, kernel, num_filters, num_colours, num_in_channels):
        super(CustomUNET, self).__init__()

        #       implementation using DownConv, UpConv and Bottlnek classes       #

    #     self.ups = nn.ModuleList()
    #     self.downs = nn.ModuleList()
    #     # Other properties if needed
    #     # Down part of the model, bottleneck, Up part of the model, final conv
    #     ##############################################################################################
    #     #                                       Your Code                                            #
    #     self.downs.append(DownConv(
    #         kernel=kernel, in_channels=in_channels, out_channels=num_filters, NF=num_filters))
    #     self.downs.append(DownConv(
    #         kernel=kernel, in_channels=num_filters, out_channels=2*num_filters, NF=2*num_filters))
    #     self.downs.append(Bottleneck(
    #         kernel=kernel, in_channels=2*num_filters, out_channels=2*num_filters, NF=2*num_filters))

    #     self.ups.append(
    #         UpConv(kernel=kernel, in_channels=2*2*num_filters, out_channels=num_filters, NF=num_filters))
    #     self.ups.append(
    #         UpConv(kernel=kernel, in_channels=2*num_filters, out_channels=num_colors, NF=num_colors))
    #     self.ups.append(nn.Conv2d(in_channels=num_colors+in_channels,
    #                                   out_channels=num_colors, kernel_size=kernel, stride=1))
    #     ##############################################################################################

    # def forward(self, x):
    #     ##############################################################################################
    #     #                                       Your Code                                            #
    #     first = self.downs[0](x)
    #     second = self.downs[1](first)
    #     third = self.downs[2](second)
    #     fourth = self.downs[1](third)
    #     fifth = self.downs[1](fourth)
    #     sixth = self.downs[1](fifth)
    #     ##############################################################################################
    #     return sixth

        #       implementation using Sequential layers inside the BaseModel class       #

        self.downconv1 = nn.Sequential(
            MyConv2d(num_in_channels, num_filters, kernel),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        self.downconv2 = nn.Sequential(
            MyConv2d(num_filters, 2*num_filters, kernel),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(2*num_filters),
            nn.ReLU()
        )

        self.bottleneck = nn.Sequential(
            MyConv2d(2*num_filters, 2*num_filters, kernel),
            nn.BatchNorm2d(2*num_filters),
            nn.ReLU()
        )

        self.upconv1 = nn.Sequential(
            MyConv2d(2*2*num_filters, num_filters, kernel),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        self.upconv2 = nn.Sequential(
            MyConv2d(2*num_filters, num_colours, kernel),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_colours),
            nn.ReLU()
        )

        self.last_conv = MyConv2d(
            num_colours+num_in_channels, num_colours, kernel)

    def forward(self, x):
        first = self.downconv1(x)
        second = self.downconv2(first)
        third = self.bottleneck(second)
        fourth = self.upconv1(torch.cat([second, third], dim=1))
        fifth = self.upconv2(torch.cat([first, fourth], dim=1))
        output = self.last_conv(torch.cat([x, fifth], dim=1))
        return output


class Residual_DownConv(nn.Module):
    def __init__(
            self, kernel, in_channels, out_channels, NF
    ):
        super(DownConv, self).__init__()

        # properties of class
        self.downconv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels, kernel_size=kernel, stride=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(NF),
            nn.ReLU()
        )

        self.downconv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels, kernel_size=kernel, stride=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(NF)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        ##############################################################################################
        #                                       Your Code                                            #
        residual = x
        x = self.downconv1(x)
        x = self.downconv2(x)
        x = x + residual
        out = self.relu(x)
        ##############################################################################################
        return out


class Residual_UpConv(nn.Module):
    def __init__(
            self, kernel, in_channels, out_channels, NF
    ):
        super(UpConv, self).__init__()

        # properties of class
        self.upconv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels, kernel_size=kernel, stride=1),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(NF),
            nn.ReLU()
        )

        self.upconv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels, kernel_size=kernel, stride=1),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(NF)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        ##############################################################################################
        #                                       Your Code                                            #
        residual = x
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = x + residual
        out = self.relu(x)
        ##############################################################################################
        return out


class Residual_Bottleneck(nn.Module):
    def __init__(
            self, kernel, in_channels, out_channels, NF
    ):
        super(Bottleneck, self).__init__()

        # properties of class
        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels, kernel_size=kernel, stride=1),
            nn.BatchNorm2d(NF),
            nn.ReLU()
        )

        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels, kernel_size=kernel, stride=1),
            nn.BatchNorm2d(NF)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        ##############################################################################################
        #                                       Your Code                                            #
        residual = x
        x = self.bottleneck1(x)
        x = self.bottleneck2()
        x = x + residual
        out = self.relu(x)
        ##############################################################################################
        return out


class Residual_CustomUNET(nn.Module):
    def __init__(self, kernel, num_filters, num_colours, num_in_channels):
        super(Residual_CustomUNET, self).__init__()

        #       implementation using Residual_DownConv, Residual_UpConv and Residual_Bottlnek classes       #

    #     self.ups = nn.ModuleList()
    #     self.downs = nn.ModuleList()
    #     # Other properties if needed
    #     # Down part of the model, bottleneck, Up part of the model, final conv
    #     ##############################################################################################
    #     #                                       Your Code                                            #
    #     self.downs.append(Residual_DownConv(
    #         kernel=kernel, in_channels=in_channels, out_channels=num_filters, NF=num_filters))
    #     self.downs.append(Residual_DownConv(
    #         kernel=kernel, in_channels=num_filters, out_channels=2*num_filters, NF=2*num_filters))
    #     self.downs.append(Residual_Bottleneck(
    #         kernel=kernel, in_channels=2*num_filters, out_channels=2*num_filters, NF=2*num_filters))

    #     self.ups.append(
    #         Residual_UpConv(kernel=kernel, in_channels=2*2*num_filters, out_channels=num_filters, NF=num_filters))
    #     self.ups.append(
    #         Residual_UpConv(kernel=kernel, in_channels=2*num_filters, out_channels=num_colors, NF=num_colors))
    #     self.ups.append(nn.Conv2d(in_channels=num_colors+in_channels,
    #                                   out_channels=num_colors, kernel_size=kernel, stride=1))
    #     ##############################################################################################

    # def forward(self, x):
    #     ##############################################################################################
    #     #                                       Your Code                                            #
    #     first = self.downs[0](x)
    #     second = self.downs[1](first)
    #     third = self.downs[2](second)
    #     fourth = self.downs[1](third)
    #     fifth = self.downs[1](fourth)
    #     sixth = self.downs[1](fifth)
    #     ##############################################################################################
    #     return sixth

        #       implementation using Sequential layers inside the BaseModel class       #

        self.downconv1_1 = nn.Sequential(
            MyConv2d(num_in_channels, num_filters, kernel),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        self.downconv1_2 = nn.Sequential(
            MyConv2d(num_in_channels, num_filters, kernel),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_filters)
        )

        self.relu = nn.ReLU()

        self.downconv2_1 = nn.Sequential(
            MyConv2d(num_filters, 2*num_filters, kernel),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(2*num_filters),
            nn.ReLU()
        )

        self.downconv2_2 = nn.Sequential(
            MyConv2d(num_filters, 2*num_filters, kernel),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(2*num_filters)
        )

        self.bottleneck_1 = nn.Sequential(
            MyConv2d(2*num_filters, 2*num_filters, kernel),
            nn.BatchNorm2d(2*num_filters),
            nn.ReLU())

        self.bottleneck_2 = nn.Sequential(
            MyConv2d(2*num_filters, 2*num_filters, kernel),
            nn.BatchNorm2d(2*num_filters)
        )

        self.upconv1_1 = nn.Sequential(
            MyConv2d(2*2*num_filters, num_filters, kernel),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        self.upconv1_2 = nn.Sequential(
            MyConv2d(2*2*num_filters, num_filters, kernel),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_filters)
        )

        self.upconv2_1 = nn.Sequential(
            MyConv2d(2*num_filters, num_colours, kernel),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_colours),
            nn.ReLU()
        )

        self.upconv2_2 = nn.Sequential(
            MyConv2d(2*num_filters, num_colours, kernel),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_colours))

        self.last_conv = MyConv2d(
            num_colours+num_in_channels, num_colours, kernel)

    def forward(self, x):
        residual = x
        first = self.downconv1_1(x)
        second = self.downconv1_2(first)
        third = second + residual
        third = self.relu(third)
        fourth = self.downconv2_1(third)
        fifth = self.downconv2_2(fourth)
        sixth = fifth + residual
        sixth = self.relu(sixth)
        seventh = self.bottleneck1(sixth)
        eighth = self.bottleneck2(seventh)
        nineth = eighth + residual
        nineth = self.relu(nineth)
        tenth = self.upconv1_1(torch.cat[eighth, nineth])
        eleventh = self.upconv1_2(torch.cat[seventh, tenth])
        twelveth = eleventh + residual
        twelveth = self.relu(torch.cat[sixth, twelveth])
        thirteenth = self.upconv2_1(torch.cat[fifth, twelveth])
        fourteenth = self.upconv2_2(torch.cat[fourth, thirteenth])
        fifteenth = fourteenth + residual
        fifteenth = self.relu(torch.cat[third, fifteenth])
        sixteenth = self.upconv2_2(torch.cat[second, fifteenth])
        seventeenth = self.upconv2_2(torch.cat[first, sixteenth])
        eighteenth = seventeenth + residual
        output = self.relu(torch.cat[x, eighteenth])

        return output

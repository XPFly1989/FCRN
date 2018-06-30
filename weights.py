import numpy as np
import torch


def load_weights(model, weights_file, dtype):

    model_params = model.state_dict()
    data_dict = np.load(weights_file, encoding='latin1').item()

    if True:
        model_params['conv1.weight'] = torch.from_numpy(data_dict['conv1']['weights']).type(dtype).permute(3,2,0,1)
        #model_params['conv1.bias'] = torch.from_numpy(data_dict['conv1']['biases']).type(dtype)
        model_params['bn1.weight'] = torch.from_numpy(data_dict['bn_conv1']['scale']).type(dtype)
        model_params['bn1.bias'] = torch.from_numpy(data_dict['bn_conv1']['offset']).type(dtype)

        model_params['layer1.0.downsample.0.weight'] = torch.from_numpy(data_dict['res2a_branch1']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer1.0.downsample.1.weight'] = torch.from_numpy(data_dict['bn2a_branch1']['scale']).type(dtype)
        model_params['layer1.0.downsample.1.bias'] = torch.from_numpy(data_dict['bn2a_branch1']['offset']).type(dtype)

        model_params['layer1.0.conv1.weight'] = torch.from_numpy(data_dict['res2a_branch2a']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer1.0.bn1.weight'] = torch.from_numpy(data_dict['bn2a_branch2a']['scale']).type(dtype)
        model_params['layer1.0.bn1.bias'] = torch.from_numpy(data_dict['bn2a_branch2a']['offset']).type(dtype)

        model_params['layer1.0.conv2.weight'] = torch.from_numpy(data_dict['res2a_branch2b']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer1.0.bn2.weight'] = torch.from_numpy(data_dict['bn2a_branch2b']['scale']).type(dtype)
        model_params['layer1.0.bn2.bias'] = torch.from_numpy(data_dict['bn2a_branch2b']['offset']).type(dtype)

        model_params['layer1.0.conv3.weight'] = torch.from_numpy(data_dict['res2a_branch2c']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer1.0.bn3.weight'] = torch.from_numpy(data_dict['bn2a_branch2c']['scale']).type(dtype)
        model_params['layer1.0.bn3.bias'] = torch.from_numpy(data_dict['bn2a_branch2c']['offset']).type(dtype)

        model_params['layer1.1.conv1.weight'] = torch.from_numpy(data_dict['res2b_branch2a']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer1.1.bn1.weight'] = torch.from_numpy(data_dict['bn2b_branch2a']['scale']).type(dtype)
        model_params['layer1.1.bn1.bias'] = torch.from_numpy(data_dict['bn2b_branch2a']['offset']).type(dtype)

        model_params['layer1.1.conv2.weight'] = torch.from_numpy(data_dict['res2b_branch2b']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer1.1.bn2.weight'] = torch.from_numpy(data_dict['bn2b_branch2b']['scale']).type(dtype)
        model_params['layer1.1.bn2.bias'] = torch.from_numpy(data_dict['bn2b_branch2b']['offset']).type(dtype)

        model_params['layer1.1.conv3.weight'] = torch.from_numpy(data_dict['res2b_branch2c']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer1.1.bn3.weight'] = torch.from_numpy(data_dict['bn2b_branch2c']['scale']).type(dtype)
        model_params['layer1.1.bn3.bias'] = torch.from_numpy(data_dict['bn2b_branch2c']['offset']).type(dtype)

        model_params['layer1.2.conv1.weight'] = torch.from_numpy(data_dict['res2c_branch2a']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer1.2.bn1.weight'] = torch.from_numpy(data_dict['bn2c_branch2a']['scale']).type(dtype)
        model_params['layer1.2.bn1.bias'] = torch.from_numpy(data_dict['bn2c_branch2a']['offset']).type(dtype)

        model_params['layer1.2.conv2.weight'] = torch.from_numpy(data_dict['res2c_branch2b']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer1.2.bn2.weight'] = torch.from_numpy(data_dict['bn2c_branch2b']['scale']).type(dtype)
        model_params['layer1.2.bn2.bias'] = torch.from_numpy(data_dict['bn2c_branch2b']['offset']).type(dtype)

        model_params['layer1.2.conv3.weight'] = torch.from_numpy(data_dict['res2c_branch2c']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer1.2.bn3.weight'] = torch.from_numpy(data_dict['bn2c_branch2c']['scale']).type(dtype)
        model_params['layer1.2.bn3.bias'] = torch.from_numpy(data_dict['bn2c_branch2c']['offset']).type(dtype)

        model_params['layer2.0.downsample.0.weight'] = torch.from_numpy(data_dict['res3a_branch1']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer2.0.downsample.1.weight'] = torch.from_numpy(data_dict['bn3a_branch1']['scale']).type(dtype)
        model_params['layer2.0.downsample.1.bias'] = torch.from_numpy(data_dict['bn3a_branch1']['offset']).type(dtype)

        model_params['layer2.0.conv1.weight'] = torch.from_numpy(data_dict['res3a_branch2a']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer2.0.bn1.weight'] = torch.from_numpy(data_dict['bn3a_branch2a']['scale']).type(dtype)
        model_params['layer2.0.bn1.bias'] = torch.from_numpy(data_dict['bn3a_branch2a']['offset']).type(dtype)

        model_params['layer2.0.conv2.weight'] = torch.from_numpy(data_dict['res3a_branch2b']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer2.0.bn2.weight'] = torch.from_numpy(data_dict['bn3a_branch2b']['scale']).type(dtype)
        model_params['layer2.0.bn2.bias'] = torch.from_numpy(data_dict['bn3a_branch2b']['offset']).type(dtype)

        model_params['layer2.0.conv3.weight'] = torch.from_numpy(data_dict['res3a_branch2c']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer2.0.bn3.weight'] = torch.from_numpy(data_dict['bn3a_branch2c']['scale']).type(dtype)
        model_params['layer2.0.bn3.bias'] = torch.from_numpy(data_dict['bn3a_branch2c']['offset']).type(dtype)

        model_params['layer2.1.conv1.weight'] = torch.from_numpy(data_dict['res3b_branch2a']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer2.1.bn1.weight'] = torch.from_numpy(data_dict['bn3b_branch2a']['scale']).type(dtype)
        model_params['layer2.1.bn1.bias'] = torch.from_numpy(data_dict['bn3b_branch2a']['offset']).type(dtype)

        model_params['layer2.1.conv2.weight'] = torch.from_numpy(data_dict['res3b_branch2b']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer2.1.bn2.weight'] = torch.from_numpy(data_dict['bn3b_branch2b']['scale']).type(dtype)
        model_params['layer2.1.bn2.bias'] = torch.from_numpy(data_dict['bn3b_branch2b']['offset']).type(dtype)

        model_params['layer2.1.conv3.weight'] = torch.from_numpy(data_dict['res3b_branch2c']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer2.1.bn3.weight'] = torch.from_numpy(data_dict['bn3b_branch2c']['scale']).type(dtype)
        model_params['layer2.1.bn3.bias'] = torch.from_numpy(data_dict['bn3b_branch2c']['offset']).type(dtype)

        model_params['layer2.2.conv1.weight'] = torch.from_numpy(data_dict['res3c_branch2a']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer2.2.bn1.weight'] = torch.from_numpy(data_dict['bn3c_branch2a']['scale']).type(dtype)
        model_params['layer2.2.bn1.bias'] = torch.from_numpy(data_dict['bn3c_branch2a']['offset']).type(dtype)

        model_params['layer2.2.conv2.weight'] = torch.from_numpy(data_dict['res3c_branch2b']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer2.2.bn2.weight'] = torch.from_numpy(data_dict['bn3c_branch2b']['scale']).type(dtype)
        model_params['layer2.2.bn2.bias'] = torch.from_numpy(data_dict['bn3c_branch2b']['offset']).type(dtype)

        model_params['layer2.2.conv3.weight'] = torch.from_numpy(data_dict['res3c_branch2c']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer2.2.bn3.weight'] = torch.from_numpy(data_dict['bn3c_branch2c']['scale']).type(dtype)
        model_params['layer2.2.bn3.bias'] = torch.from_numpy(data_dict['bn3c_branch2c']['offset']).type(dtype)

        model_params['layer2.3.conv1.weight'] = torch.from_numpy(data_dict['res3d_branch2a']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer2.3.bn1.weight'] = torch.from_numpy(data_dict['bn3d_branch2a']['scale']).type(dtype)
        model_params['layer2.3.bn1.bias'] = torch.from_numpy(data_dict['bn3d_branch2a']['offset']).type(dtype)

        model_params['layer2.3.conv2.weight'] = torch.from_numpy(data_dict['res3d_branch2b']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer2.3.bn2.weight'] = torch.from_numpy(data_dict['bn3d_branch2b']['scale']).type(dtype)
        model_params['layer2.3.bn2.bias'] = torch.from_numpy(data_dict['bn3d_branch2b']['offset']).type(dtype)

        model_params['layer2.3.conv3.weight'] = torch.from_numpy(data_dict['res3d_branch2c']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer2.3.bn3.weight'] = torch.from_numpy(data_dict['bn3d_branch2c']['scale']).type(dtype)
        model_params['layer2.3.bn3.bias'] = torch.from_numpy(data_dict['bn3d_branch2c']['offset']).type(dtype)

        model_params['layer3.0.downsample.0.weight'] = torch.from_numpy(data_dict['res4a_branch1']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer3.0.downsample.1.weight'] = torch.from_numpy(data_dict['bn4a_branch1']['scale']).type(dtype)
        model_params['layer3.0.downsample.1.bias'] = torch.from_numpy(data_dict['bn4a_branch1']['offset']).type(dtype)

        model_params['layer3.0.conv1.weight'] = torch.from_numpy(data_dict['res4a_branch2a']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer3.0.bn1.weight'] = torch.from_numpy(data_dict['bn4a_branch2a']['scale']).type(dtype)
        model_params['layer3.0.bn1.bias'] = torch.from_numpy(data_dict['bn4a_branch2a']['offset']).type(dtype)

        model_params['layer3.0.conv2.weight'] = torch.from_numpy(data_dict['res4a_branch2b']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer3.0.bn2.weight'] = torch.from_numpy(data_dict['bn4a_branch2b']['scale']).type(dtype)
        model_params['layer3.0.bn2.bias'] = torch.from_numpy(data_dict['bn4a_branch2b']['offset']).type(dtype)

        model_params['layer3.0.conv3.weight'] = torch.from_numpy(data_dict['res4a_branch2c']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer3.0.bn3.weight'] = torch.from_numpy(data_dict['bn4a_branch2c']['scale']).type(dtype)
        model_params['layer3.0.bn3.bias'] = torch.from_numpy(data_dict['bn4a_branch2c']['offset']).type(dtype)

        model_params['layer3.1.conv1.weight'] = torch.from_numpy(data_dict['res4b_branch2a']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer3.1.bn1.weight'] = torch.from_numpy(data_dict['bn4b_branch2a']['scale']).type(dtype)
        model_params['layer3.1.bn1.bias'] = torch.from_numpy(data_dict['bn4b_branch2a']['offset']).type(dtype)

        model_params['layer3.1.conv2.weight'] = torch.from_numpy(data_dict['res4b_branch2b']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer3.1.bn2.weight'] = torch.from_numpy(data_dict['bn4b_branch2b']['scale']).type(dtype)
        model_params['layer3.1.bn2.bias'] = torch.from_numpy(data_dict['bn4b_branch2b']['offset']).type(dtype)

        model_params['layer3.1.conv3.weight'] = torch.from_numpy(data_dict['res4b_branch2c']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer3.1.bn3.weight'] = torch.from_numpy(data_dict['bn4b_branch2c']['scale']).type(dtype)
        model_params['layer3.1.bn3.bias'] = torch.from_numpy(data_dict['bn4b_branch2c']['offset']).type(dtype)

        model_params['layer3.2.conv1.weight'] = torch.from_numpy(data_dict['res4c_branch2a']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer3.2.bn1.weight'] = torch.from_numpy(data_dict['bn4c_branch2a']['scale']).type(dtype)
        model_params['layer3.2.bn1.bias'] = torch.from_numpy(data_dict['bn4c_branch2a']['offset']).type(dtype)

        model_params['layer3.2.conv2.weight'] = torch.from_numpy(data_dict['res4c_branch2b']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer3.2.bn2.weight'] = torch.from_numpy(data_dict['bn4c_branch2b']['scale']).type(dtype)
        model_params['layer3.2.bn2.bias'] = torch.from_numpy(data_dict['bn4c_branch2b']['offset']).type(dtype)

        model_params['layer3.2.conv3.weight'] = torch.from_numpy(data_dict['res4c_branch2c']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer3.2.bn3.weight'] = torch.from_numpy(data_dict['bn4c_branch2c']['scale']).type(dtype)
        model_params['layer3.2.bn3.bias'] = torch.from_numpy(data_dict['bn4c_branch2c']['offset']).type(dtype)

        model_params['layer3.3.conv1.weight'] = torch.from_numpy(data_dict['res4d_branch2a']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer3.3.bn1.weight'] = torch.from_numpy(data_dict['bn4d_branch2a']['scale']).type(dtype)
        model_params['layer3.3.bn1.bias'] = torch.from_numpy(data_dict['bn4d_branch2a']['offset']).type(dtype)

        model_params['layer3.3.conv2.weight'] = torch.from_numpy(data_dict['res4d_branch2b']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer3.3.bn2.weight'] = torch.from_numpy(data_dict['bn4d_branch2b']['scale']).type(dtype)
        model_params['layer3.3.bn2.bias'] = torch.from_numpy(data_dict['bn4d_branch2b']['offset']).type(dtype)

        model_params['layer3.3.conv3.weight'] = torch.from_numpy(data_dict['res4d_branch2c']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer3.3.bn3.weight'] = torch.from_numpy(data_dict['bn4d_branch2c']['scale']).type(dtype)
        model_params['layer3.3.bn3.bias'] = torch.from_numpy(data_dict['bn4d_branch2c']['offset']).type(dtype)

        model_params['layer3.4.conv1.weight'] = torch.from_numpy(data_dict['res4e_branch2a']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer3.4.bn1.weight'] = torch.from_numpy(data_dict['bn4e_branch2a']['scale']).type(dtype)
        model_params['layer3.4.bn1.bias'] = torch.from_numpy(data_dict['bn4e_branch2a']['offset']).type(dtype)

        model_params['layer3.4.conv2.weight'] = torch.from_numpy(data_dict['res4e_branch2b']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer3.4.bn2.weight'] = torch.from_numpy(data_dict['bn4e_branch2b']['scale']).type(dtype)
        model_params['layer3.4.bn2.bias'] = torch.from_numpy(data_dict['bn4e_branch2b']['offset']).type(dtype)

        model_params['layer3.4.conv3.weight'] = torch.from_numpy(data_dict['res4e_branch2c']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer3.4.bn3.weight'] = torch.from_numpy(data_dict['bn4e_branch2c']['scale']).type(dtype)
        model_params['layer3.4.bn3.bias'] = torch.from_numpy(data_dict['bn4e_branch2c']['offset']).type(dtype)

        model_params['layer3.5.conv1.weight'] = torch.from_numpy(data_dict['res4f_branch2a']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer3.5.bn1.weight'] = torch.from_numpy(data_dict['bn4f_branch2a']['scale']).type(dtype)
        model_params['layer3.5.bn1.bias'] = torch.from_numpy(data_dict['bn4f_branch2a']['offset']).type(dtype)

        model_params['layer3.5.conv2.weight'] = torch.from_numpy(data_dict['res4f_branch2b']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer3.5.bn2.weight'] = torch.from_numpy(data_dict['bn4f_branch2b']['scale']).type(dtype)
        model_params['layer3.5.bn2.bias'] = torch.from_numpy(data_dict['bn4f_branch2b']['offset']).type(dtype)

        model_params['layer3.5.conv3.weight'] = torch.from_numpy(data_dict['res4f_branch2c']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer3.5.bn3.weight'] = torch.from_numpy(data_dict['bn4f_branch2c']['scale']).type(dtype)
        model_params['layer3.5.bn3.bias'] = torch.from_numpy(data_dict['bn4f_branch2c']['offset']).type(dtype)

        model_params['layer4.0.downsample.0.weight'] = torch.from_numpy(data_dict['res5a_branch1']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer4.0.downsample.1.weight'] = torch.from_numpy(data_dict['bn5a_branch1']['scale']).type(dtype)
        model_params['layer4.0.downsample.1.bias'] = torch.from_numpy(data_dict['bn5a_branch1']['offset']).type(dtype)

        model_params['layer4.0.conv1.weight'] = torch.from_numpy(data_dict['res5a_branch2a']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer4.0.bn1.weight'] = torch.from_numpy(data_dict['bn5a_branch2a']['scale']).type(dtype)
        model_params['layer4.0.bn1.bias'] = torch.from_numpy(data_dict['bn5a_branch2a']['offset']).type(dtype)

        model_params['layer4.0.conv2.weight'] = torch.from_numpy(data_dict['res5a_branch2b']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer4.0.bn2.weight'] = torch.from_numpy(data_dict['bn5a_branch2b']['scale']).type(dtype)
        model_params['layer4.0.bn2.bias'] = torch.from_numpy(data_dict['bn5a_branch2b']['offset']).type(dtype)

        model_params['layer4.0.conv3.weight'] = torch.from_numpy(data_dict['res5a_branch2c']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer4.0.bn3.weight'] = torch.from_numpy(data_dict['bn5a_branch2c']['scale']).type(dtype)
        model_params['layer4.0.bn3.bias'] = torch.from_numpy(data_dict['bn5a_branch2c']['offset']).type(dtype)

        model_params['layer4.1.conv1.weight'] = torch.from_numpy(data_dict['res5b_branch2a']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer4.1.bn1.weight'] = torch.from_numpy(data_dict['bn5b_branch2a']['scale']).type(dtype)
        model_params['layer4.1.bn1.bias'] = torch.from_numpy(data_dict['bn5b_branch2a']['offset']).type(dtype)

        model_params['layer4.1.conv2.weight'] = torch.from_numpy(data_dict['res5b_branch2b']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer4.1.bn2.weight'] = torch.from_numpy(data_dict['bn5b_branch2b']['scale']).type(dtype)
        model_params['layer4.1.bn2.bias'] = torch.from_numpy(data_dict['bn5b_branch2b']['offset']).type(dtype)

        model_params['layer4.1.conv3.weight'] = torch.from_numpy(data_dict['res5b_branch2c']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer4.1.bn3.weight'] = torch.from_numpy(data_dict['bn5b_branch2c']['scale']).type(dtype)
        model_params['layer4.1.bn3.bias'] = torch.from_numpy(data_dict['bn5b_branch2c']['offset']).type(dtype)

        model_params['layer4.2.conv1.weight'] = torch.from_numpy(data_dict['res5c_branch2a']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer4.2.bn1.weight'] = torch.from_numpy(data_dict['bn5c_branch2a']['scale']).type(dtype)
        model_params['layer4.2.bn1.bias'] = torch.from_numpy(data_dict['bn5c_branch2a']['offset']).type(dtype)

        model_params['layer4.2.conv2.weight'] = torch.from_numpy(data_dict['res5c_branch2b']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer4.2.bn2.weight'] = torch.from_numpy(data_dict['bn5c_branch2b']['scale']).type(dtype)
        model_params['layer4.2.bn2.bias'] = torch.from_numpy(data_dict['bn5c_branch2b']['offset']).type(dtype)

        model_params['layer4.2.conv3.weight'] = torch.from_numpy(data_dict['res5c_branch2c']['weights']).type(dtype).permute(3,2,0,1)
        model_params['layer4.2.bn3.weight'] = torch.from_numpy(data_dict['bn5c_branch2c']['scale']).type(dtype)
        model_params['layer4.2.bn3.bias'] = torch.from_numpy(data_dict['bn5c_branch2c']['offset']).type(dtype)

    model_params['conv2.weight'] = torch.from_numpy(data_dict['layer1']['weights']).type(dtype).permute(3,2,0,1)
    #model_params['conv2.bias'] = torch.from_numpy(data_dict['layer1']['biases']).type(dtype)
    model_params['bn2.weight'] = torch.from_numpy(data_dict['layer1_BN']['scale']).type(dtype)
    model_params['bn2.bias'] = torch.from_numpy(data_dict['layer1_BN']['offset']).type(dtype)

    # set True to enable weight import, or set False to initialize by yourself
    if True:

        model_params['up1.conv1_1.weight'] = torch.from_numpy(data_dict['layer2x_br1_ConvA']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up1.conv1_1.bias'] = torch.from_numpy(data_dict['layer2x_br1_ConvA']['biases']).type(dtype)

        model_params['up1.conv1_2.weight'] = torch.from_numpy(data_dict['layer2x_br1_ConvB']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up1.conv1_2.bias'] = torch.from_numpy(data_dict['layer2x_br1_ConvB']['biases']).type(dtype)

        model_params['up1.conv1_3.weight'] = torch.from_numpy(data_dict['layer2x_br1_ConvC']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up1.conv1_3.bias'] = torch.from_numpy(data_dict['layer2x_br1_ConvC']['biases']).type(dtype)

        model_params['up1.conv1_4.weight'] = torch.from_numpy(data_dict['layer2x_br1_ConvD']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up1.conv1_4.bias'] = torch.from_numpy(data_dict['layer2x_br1_ConvD']['biases']).type(dtype)

        model_params['up1.bn1_1.weight'] = torch.from_numpy(data_dict['layer2x_br1_BN']['scale']).type(dtype)
        model_params['up1.bn1_1.bias'] = torch.from_numpy(data_dict['layer2x_br1_BN']['offset']).type(dtype)

        model_params['up1.conv2_1.weight'] = torch.from_numpy(data_dict['layer2x_br2_ConvA']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up1.conv2_1.bias'] = torch.from_numpy(data_dict['layer2x_br2_ConvA']['biases']).type(dtype)

        model_params['up1.conv2_2.weight'] = torch.from_numpy(data_dict['layer2x_br2_ConvB']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up1.conv2_2.bias'] = torch.from_numpy(data_dict['layer2x_br2_ConvB']['biases']).type(dtype)

        model_params['up1.conv2_3.weight'] = torch.from_numpy(data_dict['layer2x_br2_ConvC']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up1.conv2_3.bias'] = torch.from_numpy(data_dict['layer2x_br2_ConvC']['biases']).type(dtype)

        model_params['up1.conv2_4.weight'] = torch.from_numpy(data_dict['layer2x_br2_ConvD']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up1.conv2_4.bias'] = torch.from_numpy(data_dict['layer2x_br2_ConvD']['biases']).type(dtype)

        model_params['up1.bn1_2.weight'] = torch.from_numpy(data_dict['layer2x_br2_BN']['scale']).type(dtype)
        model_params['up1.bn1_2.bias'] = torch.from_numpy(data_dict['layer2x_br2_BN']['offset']).type(dtype)

        model_params['up1.conv3.weight'] = torch.from_numpy(data_dict['layer2x_Conv']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up1.conv3.bias'] = torch.from_numpy(data_dict['layer2x_Conv']['biases']).type(dtype)

        model_params['up1.bn2.weight'] = torch.from_numpy(data_dict['layer2x_BN']['scale']).type(dtype)
        model_params['up1.bn2.bias'] = torch.from_numpy(data_dict['layer2x_BN']['offset']).type(dtype)

        model_params['up2.conv1_1.weight'] = torch.from_numpy(data_dict['layer4x_br1_ConvA']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up2.conv1_1.bias'] = torch.from_numpy(data_dict['layer4x_br1_ConvA']['biases']).type(dtype)

        model_params['up2.conv1_2.weight'] = torch.from_numpy(data_dict['layer4x_br1_ConvB']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up2.conv1_2.bias'] = torch.from_numpy(data_dict['layer4x_br1_ConvB']['biases']).type(dtype)

        model_params['up2.conv1_3.weight'] = torch.from_numpy(data_dict['layer4x_br1_ConvC']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up2.conv1_3.bias'] = torch.from_numpy(data_dict['layer4x_br1_ConvC']['biases']).type(dtype)

        model_params['up2.conv1_4.weight'] = torch.from_numpy(data_dict['layer4x_br1_ConvD']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up2.conv1_4.bias'] = torch.from_numpy(data_dict['layer4x_br1_ConvD']['biases']).type(dtype)

        model_params['up2.bn1_1.weight'] = torch.from_numpy(data_dict['layer4x_br1_BN']['scale']).type(dtype)
        model_params['up2.bn1_1.bias'] = torch.from_numpy(data_dict['layer4x_br1_BN']['offset']).type(dtype)

        model_params['up2.conv2_1.weight'] = torch.from_numpy(data_dict['layer4x_br2_ConvA']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up2.conv2_1.bias'] = torch.from_numpy(data_dict['layer4x_br2_ConvA']['biases']).type(dtype)

        model_params['up2.conv2_2.weight'] = torch.from_numpy(data_dict['layer4x_br2_ConvB']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up2.conv2_2.bias'] = torch.from_numpy(data_dict['layer4x_br2_ConvB']['biases']).type(dtype)

        model_params['up2.conv2_3.weight'] = torch.from_numpy(data_dict['layer4x_br2_ConvC']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up2.conv2_3.bias'] = torch.from_numpy(data_dict['layer4x_br2_ConvC']['biases']).type(dtype)

        model_params['up2.conv2_4.weight'] = torch.from_numpy(data_dict['layer4x_br2_ConvD']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up2.conv2_4.bias'] = torch.from_numpy(data_dict['layer4x_br2_ConvD']['biases']).type(dtype)

        model_params['up2.bn1_2.weight'] = torch.from_numpy(data_dict['layer4x_br2_BN']['scale']).type(dtype)
        model_params['up2.bn1_2.bias'] = torch.from_numpy(data_dict['layer4x_br2_BN']['offset']).type(dtype)

        model_params['up2.conv3.weight'] = torch.from_numpy(data_dict['layer4x_Conv']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up2.conv3.bias'] = torch.from_numpy(data_dict['layer4x_Conv']['biases']).type(dtype)

        model_params['up2.bn2.weight'] = torch.from_numpy(data_dict['layer4x_BN']['scale']).type(dtype)
        model_params['up2.bn2.bias'] = torch.from_numpy(data_dict['layer4x_BN']['offset']).type(dtype)

        model_params['up3.conv1_1.weight'] = torch.from_numpy(data_dict['layer8x_br1_ConvA']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up3.conv1_1.bias'] = torch.from_numpy(data_dict['layer8x_br1_ConvA']['biases']).type(dtype)

        model_params['up3.conv1_2.weight'] = torch.from_numpy(data_dict['layer8x_br1_ConvB']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up3.conv1_2.bias'] = torch.from_numpy(data_dict['layer8x_br1_ConvB']['biases']).type(dtype)

        model_params['up3.conv1_3.weight'] = torch.from_numpy(data_dict['layer8x_br1_ConvC']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up3.conv1_3.bias'] = torch.from_numpy(data_dict['layer8x_br1_ConvC']['biases']).type(dtype)

        model_params['up3.conv1_4.weight'] = torch.from_numpy(data_dict['layer8x_br1_ConvD']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up3.conv1_4.bias'] = torch.from_numpy(data_dict['layer8x_br1_ConvD']['biases']).type(dtype)

        model_params['up3.bn1_1.weight'] = torch.from_numpy(data_dict['layer8x_br1_BN']['scale']).type(dtype)
        model_params['up3.bn1_1.bias'] = torch.from_numpy(data_dict['layer8x_br1_BN']['offset']).type(dtype)

        model_params['up3.conv2_1.weight'] = torch.from_numpy(data_dict['layer8x_br2_ConvA']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up3.conv2_1.bias'] = torch.from_numpy(data_dict['layer8x_br2_ConvA']['biases']).type(dtype)

        model_params['up3.conv2_2.weight'] = torch.from_numpy(data_dict['layer8x_br2_ConvB']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up3.conv2_2.bias'] = torch.from_numpy(data_dict['layer8x_br2_ConvB']['biases']).type(dtype)

        model_params['up3.conv2_3.weight'] = torch.from_numpy(data_dict['layer8x_br2_ConvC']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up3.conv2_3.bias'] = torch.from_numpy(data_dict['layer8x_br2_ConvC']['biases']).type(dtype)

        model_params['up3.conv2_4.weight'] = torch.from_numpy(data_dict['layer8x_br2_ConvD']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up3.conv2_4.bias'] = torch.from_numpy(data_dict['layer8x_br2_ConvD']['biases']).type(dtype)

        model_params['up3.bn1_2.weight'] = torch.from_numpy(data_dict['layer8x_br2_BN']['scale']).type(dtype)
        model_params['up3.bn1_2.bias'] = torch.from_numpy(data_dict['layer8x_br2_BN']['offset']).type(dtype)

        model_params['up3.conv3.weight'] = torch.from_numpy(data_dict['layer8x_Conv']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up3.conv3.bias'] = torch.from_numpy(data_dict['layer8x_Conv']['biases']).type(dtype)

        model_params['up3.bn2.weight'] = torch.from_numpy(data_dict['layer8x_BN']['scale']).type(dtype)
        model_params['up3.bn2.bias'] = torch.from_numpy(data_dict['layer8x_BN']['offset']).type(dtype)

        model_params['up4.conv1_1.weight'] = torch.from_numpy(data_dict['layer16x_br1_ConvA']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up4.conv1_1.bias'] = torch.from_numpy(data_dict['layer16x_br1_ConvA']['biases']).type(dtype)

        model_params['up4.conv1_2.weight'] = torch.from_numpy(data_dict['layer16x_br1_ConvB']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up4.conv1_2.bias'] = torch.from_numpy(data_dict['layer16x_br1_ConvB']['biases']).type(dtype)

        model_params['up4.conv1_3.weight'] = torch.from_numpy(data_dict['layer16x_br1_ConvC']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up4.conv1_3.bias'] = torch.from_numpy(data_dict['layer16x_br1_ConvC']['biases']).type(dtype)

        model_params['up4.conv1_4.weight'] = torch.from_numpy(data_dict['layer16x_br1_ConvD']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up4.conv1_4.bias'] = torch.from_numpy(data_dict['layer16x_br1_ConvD']['biases']).type(dtype)

        model_params['up4.bn1_1.weight'] = torch.from_numpy(data_dict['layer16x_br1_BN']['scale']).type(dtype)
        model_params['up4.bn1_1.bias'] = torch.from_numpy(data_dict['layer16x_br1_BN']['offset']).type(dtype)

        model_params['up4.conv2_1.weight'] = torch.from_numpy(data_dict['layer16x_br2_ConvA']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up4.conv2_1.bias'] = torch.from_numpy(data_dict['layer16x_br2_ConvA']['biases']).type(dtype)

        model_params['up4.conv2_2.weight'] = torch.from_numpy(data_dict['layer16x_br2_ConvB']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up4.conv2_2.bias'] = torch.from_numpy(data_dict['layer16x_br2_ConvB']['biases']).type(dtype)

        model_params['up4.conv2_3.weight'] = torch.from_numpy(data_dict['layer16x_br2_ConvC']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up4.conv2_3.bias'] = torch.from_numpy(data_dict['layer16x_br2_ConvC']['biases']).type(dtype)

        model_params['up4.conv2_4.weight'] = torch.from_numpy(data_dict['layer16x_br2_ConvD']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up4.conv2_4.bias'] = torch.from_numpy(data_dict['layer16x_br2_ConvD']['biases']).type(dtype)

        model_params['up4.bn1_2.weight'] = torch.from_numpy(data_dict['layer16x_br2_BN']['scale']).type(dtype)
        model_params['up4.bn1_2.bias'] = torch.from_numpy(data_dict['layer16x_br2_BN']['offset']).type(dtype)

        model_params['up4.conv3.weight'] = torch.from_numpy(data_dict['layer16x_Conv']['weights']).type(dtype).permute(3,2,0,1)
        model_params['up4.conv3.bias'] = torch.from_numpy(data_dict['layer16x_Conv']['biases']).type(dtype)

        model_params['up4.bn2.weight'] = torch.from_numpy(data_dict['layer16x_BN']['scale']).type(dtype)
        model_params['up4.bn2.bias'] = torch.from_numpy(data_dict['layer16x_BN']['offset']).type(dtype)

        model_params['conv3.weight'] = torch.from_numpy(data_dict['ConvPred']['weights']).type(dtype).permute(3,2,0,1)
        model_params['conv3.bias'] = torch.from_numpy(data_dict['ConvPred']['biases']).type(dtype)

    print('Up proj weights loaded!!!!!!!!!!!')
    return model_params

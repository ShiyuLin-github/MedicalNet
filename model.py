import torch
from torch import nn
from models import resnet


def generate_model(opt):
    assert opt.model in [
        'resnet'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        
        if opt.model_depth == 10:
            model = resnet.resnet10(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
    
    if not opt.no_cuda:
        if len(opt.gpu_id) > 1:
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=opt.gpu_id)
            net_dict = model.state_dict() 
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu_id[0])
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()
    
    # load pretrain
    if opt.phase != 'test' and opt.pretrain_path:
        print ('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
        #这行代码实际上是将预训练模型的状态字典中与当前模型参数匹配的部分提取出来，以便进行参数加载。
        # pretrain['state_dict'].items(): 这部分迭代了预训练模型状态字典的键值对，其中 pretrain['state_dict'] 是预训练模型的状态字典。
        # if k in net_dict.keys(): 这个条件检查当前键 k 是否存在于当前模型的状态字典 net_dict 的键集合中。
        # {k: v for k, v in ...}: 这个构造式创建了一个新的字典，其中只包含满足条件的键值对。这些键值对是预训练模型状态字典中与当前模型状态字典匹配的部分。
        # 最终，pretrain_dict 将只包含与当前模型匹配的键值对，用于加载预训练模型的权重到当前模型。这样做是因为在迁移学习或微调任务中，通常只加载与当前模型架构匹配的参数。不匹配的参数将被忽略，或者根据需要进行初始化。这样可以避免由于模型架构不同导致的参数维度不匹配的问题。
         
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = [] 
        for pname, p in model.named_parameters(): #named_parameters() 方法返回一个迭代器，其中包含每个参数的名称（pname）和参数张量（p）。
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters, 
                      'new_parameters': new_parameters}

        return model, parameters

    return model, model.parameters()

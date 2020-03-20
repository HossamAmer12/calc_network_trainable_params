clc
clear
selector = 0;

% Summary:
% ---------
% inceptionv3
% Total number of trainable parameters: 23.8164M

% resnet50
% Total number of trainable parameters: 25.5295M

% inceptionresnetv2
% Total number of trainable parameters: 55.8122M

if selector == 0
    net = inceptionv3;
    disp('inceptionv3:')
else
    if selector == 1
     net = resnet50;
     disp('resnet50:')
    else 
        net = inceptionresnetv2;
        disp('inceptionresnetv2:')
    end
end

layers = net.Layers;

% Total number of parameters:
total_params = 0;

% Skip the input
for i=1:length(layers)
    name = layers(i).Name;
    
    if contains(name, "conv2d")  ...
       || contains(name, "res")  ...
       || contains(name, "conv")
        
        if contains(name, 'bn') || contains(name, 'ac')
            continue;
        end
        
        w = layers(i).FilterSize(1);
        h = layers(i).FilterSize(2);
        c = layers(i).NumChannels;
        k = layers(i).NumFilters;
        total_params = total_params + (w * h * c * k) + (1 * 1 * k);
    end
    
    if contains(name, "max_pooling") 
        total_params = total_params + 0;
    end
    
    if contains(name, "avg_pool") 
        total_params = total_params + 0;
    end
    
    if contains(name, "predictions") || contains(name, 'fc1000')
        current_layer_n = layers(i).OutputSize;
        prev_layer_n    = layers(i).InputSize;
        total_params    = total_params + ((current_layer_n*prev_layer_n)+1);
        break; 
    end

end

disp(['- Total number of trainable parameters: ' ...
    num2str(total_params/10^6) 'M']);

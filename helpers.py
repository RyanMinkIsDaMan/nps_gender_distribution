import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import torch

def tokenize_name(text):
    return list(text.lower())

def numericalize_name(tokenized_name, vocab_to_int_map):
    return [vocab_to_int_map[token] for token in tokenized_name]

def pad_features(numericalized_names, sequence_length):
    features = np.zeros((len(numericalized_names), sequence_length), dtype=int)
    
    for i, name in enumerate(numericalized_names):
        features[i, -len(name):] = np.array(name)[:sequence_length]
        
    return features

def plot_confusion_matrix(cm):
    labels = ['female', 'male']
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(10,7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    
def get_model_performance(classifier, data_loader, is_training_on_gpu):
    classifier.eval()

    total_num_correct = 0
    all_outputs = None
    all_labels = None

    for inputs, labels in data_loader:
        if is_training_on_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        # reset hidden states
        hidden_states = classifier.init_hidden_states(inputs.shape[0])

        # get the outputs from the model
        outputs, _ = classifier(inputs, hidden_states)
        outputs = outputs.squeeze().round()

        # get how many of the predictions are correct
        correct_tensor = outputs.eq(labels.float().view_as(outputs))
        num_correct = correct_tensor.squeeze().cpu().numpy() if is_training_on_gpu else correct_tensor.squeeze().numpy()
        total_num_correct += np.sum(num_correct)

        outputs = outputs.squeeze().round().int().cpu().detach().numpy()
        labels = labels.cpu().numpy()

        # accumulate outputs
        if all_outputs is None:
            all_outputs = outputs
        else:
            all_outputs = np.concatenate((all_outputs, outputs))

        # accumulate labels
        if all_labels is None:
            all_labels = labels
        else:
            all_labels = np.concatenate((all_labels, labels))

    # calculate accuracy
    accuracy = total_num_correct / len(data_loader.dataset)
    
    return accuracy, all_outputs, all_labels

def infer_gender_from_name(name, sequence_length, vocab_to_int_map, classifier, is_training_on_gpu):
    # preprocess name
    tokenized_name = tokenize_name(name)
    numericalized_name = numericalize_name(tokenized_name, vocab_to_int_map)
    features = pad_features([numericalized_name], sequence_length)
    
    # prepare model
    classifier.eval()
    hidden_states = classifier.init_hidden_states(1)
    feature_tensor = torch.from_numpy(features)
    
    if is_training_on_gpu:
        feature_tensor = feature_tensor.cuda()
     
    # forward pass the name to the model
    output, hidden_states = classifier(feature_tensor.long(), hidden_states)
    output = output.squeeze().round()
    
    gender = 'female' if output.item() == 0 else 'male'
    
    return gender
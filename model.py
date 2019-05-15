import torch
import torch.nn as nn

is_training_on_gpu = torch.cuda.is_available()

class Classifier(nn.Module):
    def __init__(self, sequence_length, num_layers, vocab_size, hidden_dim, output_size):
        '''
        Define model architecture
        '''
        super(Classifier, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        
        self.one_hot_matrix = torch.eye(vocab_size)
        self.lstm = nn.LSTM(vocab_size, hidden_dim, num_layers=num_layers, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(p=0.5)
        
    def one_hot_encode(self, labels):
        '''
        Embedding labels to one-hot form
        '''
        if is_training_on_gpu:
            return self.one_hot_matrix[labels].cuda()
        else:
            return self.one_hot_matrix[labels]
        
    def init_hidden_states(self, batch_size):
        '''
        Initialize hidden and cell states
        '''
        weights = next(self.parameters()).data
        
        if is_training_on_gpu:
            hidden_states = (weights.new(self.num_layers, batch_size, self.hidden_dim).zero_().cuda(), 
                             weights.new(self.num_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden_states = (weights.new(self.num_layers, batch_size, self.hidden_dim).zero_(), 
                             weights.new(self.num_layers, batch_size, self.hidden_dim).zero_())
            
        return hidden_states
    
    def forward(self, input, hidden_states):
        '''
        Define how data flows from one layer to the next
        '''
        batch_size = input.shape[0]
        
        one_hot_embedding = self.one_hot_encode(input)
        lstm_output, hidden_states = self.lstm(one_hot_embedding, hidden_states)
        lstm_output = lstm_output.contiguous().view(-1, self.hidden_dim)
        lstm_output = self.dropout(lstm_output)
        output = self.fc(lstm_output)
        output = torch.sigmoid(output)
        output = output.view(batch_size, -1)
        output = output[:, -1]
        
        return output, hidden_states
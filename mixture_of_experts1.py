import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

from embeddings import Embeddings

class MoEModel(nn.Module):
    def __init__(self, seq_len, embed_size, experts, num_experts, config, dropout_rate=0.2):
        super(MoEModel, self).__init__()

        self.seq_len = seq_len
        self.embed_size = embed_size
        self.hidden_size = (seq_len * 2 // 3) + 4
        self.experts = experts
        self.num_experts = num_experts
        self.config = config
        self.dropout = nn.Dropout(dropout_rate)

        self.embeddings = Embeddings(self.config)



        self.weights_1 = nn.Parameter(torch.zeros(self.embed_size * self.seq_len, self.hidden_size))
        nn.init.xavier_uniform_(self.weights_1)
        #self.bias_1 = nn.Parameter(torch.zeros(self.hidden_size))
        #nn.init.uniform_(self.bias_1)

        self.weights_2 = nn.Parameter(torch.zeros(self.hidden_size, self.num_experts))
        nn.init.xavier_uniform_(self.weights_2)
        #self.bias_2 = nn.Parameter(torch.zeros(self.num_experts))
        #nn.init.uniform_(self.bias_2)

    def embedding_lookup(self, input_ids):
        return self.embeddings(input_ids).view(input_ids.shape[0], -1)

    def forward(self, x, attention_mask, start_positions=None, end_positions=None):

        #print(x)
        x_embed = self.embedding_lookup(x) # b x (384 * embed_size)
        #print(x_embed)
        logits = x_embed @ self.weights_1 # b x hidden_size
        #logits += self.bias_1 # b x hidden_size
        logits = F.relu(logits) # b x hidden_size
        logits = self.dropout(logits)
        logits = logits @ self.weights_2 # b x output_size
        #logits += self.bias_2 # b x output_size
        logits = F.relu(logits) # b x output_size
        p = F.softmax(logits, 1) # b x 4
        #print(p)

        loss = None
        start_logits_ave = None
        end_logits_ave = None

        b_size = p.shape[0]
        for i in range(self.num_experts):
            expert_output = self.experts[i](x, attention_mask=attention_mask, start_positions=start_positions,
                                            end_positions=end_positions, return_dict=True)

            logit_p = torch.reshape(p[:, i], (b_size, 1)).clone().expand(b_size, 384)
            start_logits = expert_output.start_logits
            end_logits = expert_output.end_logits

            

            if start_logits_ave is None:
                start_logits_ave = logit_p * start_logits
            else:
                start_logits_ave += logit_p * start_logits
            if end_logits_ave is None:
                end_logits_ave = logit_p * end_logits
            else:
                end_logits_ave += logit_p * end_logits

                

            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index, reduction='none')
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                if loss is None:
                    loss = p[:,i] * ((start_loss + end_loss) / 2)
                else:
                    loss += p[:,i] * ((start_loss + end_loss) / 2)

        
            """
                start_logits_loss = F.softmax(start_logits, 1)
                end_logits_loss = F.softmax(end_logits, 1)

                start_loss = 0
                end_loss = 0

                batch_loss = torch.zeros(b_size, device='cuda:0', requires_grad=False)
                for b in range(b_size):
                    start_loss += 1 - start_logits_loss[b,start_positions[b]]
                    end_loss += 1 - end_logits_loss[b, end_positions[b]]
                    batch_loss[b] = (start_loss + end_loss) / 2
                if loss is None:                                                                                                      
                    loss = p[:,i] * batch_loss                                                                     
                else:
                    loss += p[:,i] * batch_loss
            """
        """
        if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension                                                                        
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms                                
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits_ave, start_positions)
                end_loss = loss_fct(end_logits_ave, end_positions)
                loss = (start_loss + end_loss) / 2
        """

        if start_positions is not None and end_positions is not None:
            loss = torch.sum(loss) / b_size
                    
        return loss, start_logits_ave, end_logits_ave

        """
        loss = None

        first_output = self.experts[0](x, attention_mask=attention_mask, start_positions=start_positions, 
            end_positions=end_positions, return_dict=True)

        if not (start_positions is None or end_positions is None):
            first_loss = first_output.loss
            #first_loss = torch.exp(first_loss / 2)
            loss = p[:,0] * first_loss

        b_size = p.shape[0]
        start_logit_p =  torch.reshape(p[:,0], (b_size, 1)).clone().expand(b_size, 384)
        start_logits = start_logit_p * first_output.start_logits
        end_logits = start_logit_p * first_output.end_logits

        for i in range(1, self.num_experts):
            expert_output = self.experts[i](x, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions, return_dict=True)
            if not (start_positions is None or end_positions is None):
                expert_loss = expert_output.loss
                #expert_loss = torch.exp(expert_loss / 2)
                loss = loss + p[:,i] * expert_loss
            logit_p = torch.reshape(p[:,0], (b_size, 1)).clone().expand(b_size, 384)
            start_logits = start_logits + logit_p * expert_output.start_logits
            end_logits = end_logits + logit_p * expert_output.end_logits
        if not (start_positions is None or end_positions is None):
            #loss = -torch.log(loss)
            loss = torch.sum(loss)
        return loss, start_logits, end_logits
        """
        
    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = MoEModel(experts=params['experts'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save_pretrained(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """

        params = {
            'args': dict(seq_len=self.seq_len, embed_size=self.embed_size,
                         num_experts=self.num_experts, config=self.config),
            'experts': self.experts,
            'state_dict': self.state_dict()
        }

        output_model_file = os.path.join(path, 'moe_model_params')

        torch.save(params, output_model_file)

from prettytable import PrettyTable

class Utils:

    @classmethod
    def print_metrics(cls,avg_loss, avg_acc):
        t = PrettyTable(['Parameter', 'Value'])
        t.add_row(['Avg_Loss', avg_loss])
        t.add_row(['Avg_Acc', avg_acc])
        print(t)
        return
    
    
    @classmethod
    def shut_down_decoder(cls,model,flag):
        i=0
        for parameter in model.parameters():
            parameter.requires_grad = flag
            i+=1
            if(i==108):
                break
        i=0
        for parameter in model.parameters():
            if parameter.requires_grad :
                print(i,"True",parameter.data.shape)
            else:
                print(i,"False",parameter.shape)
            i+=1
        return 
    
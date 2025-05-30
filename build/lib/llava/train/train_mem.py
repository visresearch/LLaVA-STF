from llava.train.train import train_th

if __name__ == "__main__":
    # print("------------------------------------------------------------------------------")
    train_th(attn_implementation="flash_attention_2")

from llava.train.train import train

if __name__ == "__main__":
    # print("------------------------------------------------------------------------------")
    train(attn_implementation="flash_attention_2")

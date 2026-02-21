import matplotlib.pyplot as plt
import numpy as np

def param_graph():
    # Map num transformer blocks to num parameters
    data = {
         1: 2531318,
         2: 2729590,
         4: 3126134,
         8: 3919222,
        16: 5505398,
    }

    blocks = list(data.keys())
    params = list(data.values())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(blocks, params, marker="o")

    # Titles and labels
    ax.set_title("Parameters vs Transformer Blocks", fontsize=14)
    ax.set_xlabel("Number of transformer blocks", fontsize=12)
    ax.set_ylabel("Number of parameters", fontsize=12)

    # Tick label sizes
    ax.tick_params(axis="both", labelsize=12)

    # Make the '1e6' offset text match tick label size
    ax.yaxis.get_offset_text().set_fontsize(12)

    ax.grid(True)
    fig.tight_layout()
    plt.show()

def loss_hist():
    # Data
    depths = [1, 2, 4, 8, 16]
    train_loss = [0.8750, 0.7087, 0.6271, 0.7452, 1.6911]
    dev_loss   = [2.4693, 2.6420, 2.7276, 2.9093, 2.8123]
    test_loss  = [2.4024, 2.5446, 2.6532, 2.8269, 2.7525]

    plt.figure()
    plt.plot(depths, train_loss, marker="o", label="Train")
    plt.plot(depths, dev_loss,   marker="o", label="Dev")
    plt.plot(depths, test_loss,  marker="o", label="Test")
    plt.xlabel("d_model")
    plt.ylabel("Loss")
    plt.title("Train, dev, and test loss vs model depth")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    #x = np.arange(len(depths))          # positions for each depth
    #width = 0.25                        # width of each bar

    #fig, ax = plt.subplots()

    #ax.bar(x - width, train_loss, width, label="Train")
    #ax.bar(x,         dev_loss,   width, label="Dev")
    #ax.bar(x + width, test_loss,  width, label="Test")

    ## Tick label sizes
    #ax.tick_params(axis="both", labelsize=12)

    ## Make the '1e6' offset text match tick label size
    #ax.yaxis.get_offset_text().set_fontsize(12)

    #ax.set_xlabel("Depth", fontsize=14)
    #ax.set_ylabel("Loss", fontsize=14)
    #ax.set_title("Training, Dev, and Test Loss by Model Depth", fontsize=14)
    #ax.set_xticks(x)
    #ax.set_xticklabels(depths)
    #ax.legend(ncol=3)

    #fig.tight_layout()
    #plt.show()


def width_param():
    d_model = [128, 256, 512, 1024]
    params = [2531318, 5446774, 12457334, 31197046]

    plt.figure()
    plt.plot(d_model, params, marker="o")
    plt.xlabel("d_model")
    plt.ylabel("Parameters")
    plt.title("Parameter count vs model width (d_model)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def width_loss():
    d_model = [128, 256, 512, 1024]
    train_loss = [0.8681, 0.2587, 0.1330, 0.1435]
    dev_loss   = [2.5311, 2.6050, 2.6428, 2.6671]
    test_loss  = [2.4472, 2.5178, 2.5709, 2.5516]

    plt.figure()
    plt.plot(d_model, train_loss, marker="o", label="Train")
    plt.plot(d_model, dev_loss,   marker="o", label="Dev")
    plt.plot(d_model, test_loss,  marker="o", label="Test")
    plt.xlabel("d_model")
    plt.ylabel("Loss")
    plt.title("Train, dev, and test loss vs model width (d_model)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



#param_graph()
loss_hist()
#width_param()
#width_loss()

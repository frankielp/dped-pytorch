import sys


def train_args(arguments):
    """
    Parse argument for training
    """
    # specifying default parameters
    params = {
        "batch_size": 50,
        "train_size": 30000,
        "lr": 5e-4,
        "epochs": 20000,
        "w_content": 10,
        "w_color": 0.5,
        "w_texture": 1,
        "w_tv": 2000,
        "datadir": "dped/",
        "vgg_pretrained": "vgg_pretrained/imagenet-vgg-verydeep-19.mat",
        "eval_step": 1000,
        "phone": "",
    }
    for args in arguments:
        if args.startswith("model"):
            params["phone"] = args.split("=")[1]

        if args.startswith("batch_size"):
            params["batch_size"] = int(args.split("=")[1])

        if args.startswith("train_size"):
            params["train_size"] = int(args.split("=")[1])

        if args.startswith("lr"):
            params["lr"] = float(args.split("=")[1])

        if args.startswith("epochs"):
            params["epochs"] = int(args.split("=")[1])

        # -----------------------------------

        if args.startswith("w_content"):
            params["w_content"] = float(args.split("=")[1])

        if args.startswith("w_color"):
            params["w_color"] = float(args.split("=")[1])

        if args.startswith("w_texture"):
            params["w_texture"] = float(args.split("=")[1])

        if args.startswith("w_tv"):
            params["w_tv"] = float(args.split("=")[1])

        # -----------------------------------

        if args.startswith("datadir"):
            params["datadir"] = args.split("=")[1]

        if args.startswith("vgg_pretrained"):
            params["vgg_pretrained"] = args.split("=")[1]

        if args.startswith("eval_step"):
            params["eval_step"] = int(args.split("=")[1])

    if params["phone"] not in ["iphone", "sony", "blackberry"]:
        print("\nPlease specify the correct camera model:\n")
        print("python train_model.py model={iphone,blackberry,sony}\n")
        sys.exit()

    print("\nThe following parameters will be applied for CNN training:\n")

    print("Phone model:", params["phone"])
    print("Batch size:", params["batch_size"])
    print("Learning rate:", params["lr"])
    print("Epochs:", params["epochs"])
    print()
    print("Content loss:", params["w_content"])
    print("Color loss:", params["w_color"])
    print("Texture loss:", params["w_texture"])
    print("Total variation loss:", params["w_tv"])
    print()
    print("Path to DPED dataset:", params["datadir"])
    print("Path to VGG-19 network:", params["vgg_pretrained"])
    print("Evaluation step:", params["eval_step"])
    print()
    return params


def test_args(arguments):
    """
    Parse argument for inference
    """

    params = {
        "phone": "",
        "datadir": "dped/",
        "test_subset": "small",
        "iteration": "all",
        "resolution": "orig",
        "use_gpu": "true",
    }

    for args in arguments:
        if args.startswith("model"):
            params["phone"] = args.split("=")[1]

        if args.startswith("datadir"):
            params["datadir"] = args.split("=")[1]

        if args.startswith("test_subset"):
            params["test_subset"] = args.split("=")[1]

        if args.startswith("iteration"):
            params["iteration"] = args.split("=")[1]

        if args.startswith("resolution"):
            params["resolution"] = args.split("=")[1]

        if args.startswith("use_gpu"):
            params["use_gpu"] = args.split("=")[1]

    if params["phone"] == "":
        print(
            "\nPlease specify the model by running the script with the following parameter:\n"
        )
        print(
            "python test_model.py model={iphone,blackberry,sony,iphone_orig,blackberry_orig,sony_orig}\n"
        )
        sys.exit()

    return params

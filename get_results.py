import subprocess

command_ = "python eval.py --c config/{}/{}/{}_{}_lb{}_s0.yaml"

datasets_type = [
    "audio",
    # "nlp",
    # "classic_cv",
]

datasets = {}
datasets["audio"] = [
    # "bvcc",
    "vcc2018",
]
datasets["nlp"] = [
    "yelp_review",
]
datasets["classic_cv"] = [
    "utkface",
]

methods = [
    "supervised",
    "pimodel",
    "meanteacher",
    "clss",
    "ucvme",
    "mixmatch",
    "rankup",
    "rankuprda",
    # "rcus",
]

num_lbs = [
    10,
    50,
    250,
    2000,
]

with open("results.log", "w") as f:
    for dtype in datasets_type:
        for dataset in datasets[dtype]:
            for method in methods:
                for num_lb in num_lbs:
                    command = command_.format(dtype, method, method, dataset, num_lb)
                    result = subprocess.run(command, shell=True, stdout=f, stderr=subprocess.STDOUT)

                    if result.returncode == 0:
                        pass
                    else:
                        print("{} {} {} {} ERROR {}".format(dtype, method, dataset, num_lb, result.returncode))

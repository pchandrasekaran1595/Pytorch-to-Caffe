import sys

from torch import rand
from models import get_model
from pytorch2caffe import pytorch2caffe


def main():
    name = "mobilenet-v3-small"
    size = 224

    args_1 = "--name"
    args_2 = "--size"

    if args_1 in sys.argv: name = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: size = int(sys.argv[sys.argv.index(args_2) + 1])

    model = get_model(name)

    dummy = rand(1, 3, size, size)

    pytorch2caffe.trans_net(model, dummy, name) 
    pytorch2caffe.save_prototxt("./op/{}.prototxt".format(name)) 
    pytorch2caffe.save_caffemodel("./op/{}.caffemodel".format(name))


if __name__ == "__main__":
    sys.exit(main() or 0)
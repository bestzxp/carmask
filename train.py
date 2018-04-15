from unet import Unet
from utils import read_data, read_mask_img
import tensorflow as tf
import argparse
import os


def build_parser():
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument('--img_width', type=int, default=960)
    parser.add_argument('--img_height', type=int, default=640)
    parser.add_argument('--filter_num', type=int, default=64)
    parser.add_argument('--batch_norm', action='store_true', default=False)

    # checkpoint path
    parser.add_argument('--checkpoint_path', type=str, default=None)

    # train data
    parser.add_argument('--train_dir', type=str, default='./data/train')
    parser.add_argument('--train_mask_dir', type=str,
                        default='./data/train_masks')
    parser.add_argument('--val_dir', type=str, default='./data/val')
    parser.add_argument('--val_mask_dir', type=str, default='./data/val_masks')
    parser.add_argument('--n_images', type=int, default=0)

    # train parameters
    parser.add_argument('--dice_loss', action='store_true', default=False)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--always_save', action='store_true', default=False)

    return parser


def main():
    args = build_parser().parse_args()
    image_size = [args.img_height, args.img_width]
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 1.0
    # sess = tf.Session(config=config)
    sess = tf.Session()
    unet = Unet(input_shape=image_size, sess=sess, filter_num=args.filter_num, batch_norm=args.batch_norm)
    unet.build_net()
    if args.checkpoint_path:
        unet.load_weights(args.checkpoint_path)

    images = os.listdir(args.train_dir)
    images.sort()
    images = [os.path.join(args.train_dir, img) for img in images]

    masks = os.listdir(args.train_mask_dir)
    masks.sort()
    masks = [os.path.join(args.train_mask_dir, img) for img in masks]
    
    val_images = os.listdir(args.val_dir)
    val_images.sort()
    val_images = [os.path.join(args.val_dir, img) for img in val_images]


    val_masks = os.listdir(args.val_mask_dir)
    val_masks.sort()
    val_masks = [os.path.join(args.val_mask_dir, img) for img in val_masks]
    print(val_masks[0])
    # images, masks = read_data(args.train_dir,
    #                           args.train_mask_dir,
    #                           n_images=args.n_images, image_size=image_size)
    # val_images, val_masks = read_data(args.val_dir,
    #                                   args.val_mask_dir,
    #                                   n_images=args.n_images // 4, image_size=image_size)
    unet.train(images=images, masks=masks, val_images=val_images, val_masks=val_masks, epochs=args.epochs,
               batch_size=args.batch_size, learning_rate=args.learning_rate, dice_loss=args.dice_loss,
               always_save=args.always_save, image_size=image_size)


if __name__ == '__main__':
    main()
    # import cv2
    # img = read_mask_img('./data/val_masks/00087a6bd4dc_03_mask.gif')
    # print(img.shape)
    # cv2.imshow('frame', img)
    # cv2.waitKey(3000)

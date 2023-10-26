# Active Predictive Coding Vision Model
## Example usage:
* Train: `python3 train.py --dataset mnist --outdir apc_mnist --epochs 1500`
* Evaluate: `python3 train.py --dataset mnist --outdir apc_mnist --eval 1`
* Visualize: `python3 visualize.py --dataset mnist --outdir apc_mnist`

## Spatial transformer
* Big thanks to Kevin Zakka for the spatial transformer module in `transformer.py` taken from: [https://github.com/kevinzakka/spatial-transformer-network/tree/master](https://github.com/kevinzakka/spatial-transformer-network/tree/master) and modified.
* The Omniglot dataset was taken from: [https://github.com/brendenlake/omniglot](https://github.com/brendenlake/omniglot)

## Examples of parsing strategies (MNIST)
![](https://raw.githubusercontent.com/gklezd/apc-vision/main/examples/example1.gif)
![](https://raw.githubusercontent.com/gklezd/apc-vision/main/examples/example2.gif)
![](https://raw.githubusercontent.com/gklezd/apc-vision/main/examples/example3.gif)
![](https://raw.githubusercontent.com/gklezd/apc-vision/main/examples/example4.gif)

## Notes:
* This model runs on Tensorflow 2.
* You will need to unzip the Omniglot data in to the `data` folder.

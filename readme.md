### DataPaneer

##### DefHacks 2020 Submission | [DevPost](https://devpost.com/software/datapaneer)| [YouTube](https://www.youtube.com/watch?v=IybTefxD1sc) |

<b>Collaborators</b>: [Vikram Sambamurthy](https://github.com/v97), [Siddhartha Datta](https://github.com/dattasiddhartha/)

<img src="./static/webapp.PNG" height="250px"></img>
<img src="./static/inversecooking.PNG" height="250px"></img>

Roadmap to this product
* Scraped food data (image, ingredients, health information)
* Recipe recommendation engine
* Food style/variations generation (food style transfer)
* Recipe generation (image → list of ingredients reverse engineering)
* Web app

#### Weights and Data

Weights can be downloaded from [here](https://drive.google.com/drive/folders/1Suq1pMC7chu1uKcS_vpeQEh0g2og8WBM?usp=sharing).

Data for training CycleGAN: [[food]](https://github.com/karansikka1/iFood_2019)

#### Food Style Transfer

Style weights stored in './fast_neural_style_transfer/models'.

#### CycleGAN

Place data in `./datasets/` with image set pairs as `trainA` and `trainB` for training. Training time can go up to 13hrs per food mask, tested on GTX1070.

##### Development server

Run python server using `python app.py` command

### DataPaneer

##### DefHacks 2020 Submission

<b>Collaborators</b>: [Vikram Sambamurthy](https://github.com/v97), [Siddhartha Datta](https://github.com/dattasiddhartha/)

<!--Image-->

Roadmap to this product
* Scraped food data (image, ingredients, health information)
* Recipe recommendation engine
* Food style/variations generation (food style transfer)
* Recipe generation (image → list of ingredients reverse engineering)
* Web app

#### Weights and Data

Weights can be downloaded from [here](https://drive.google.com/drive/folders/1Suq1pMC7chu1uKcS_vpeQEh0g2og8WBM?usp=sharing).

Data for training CycleGAN: [[food]](https://github.com/karansikka1/iFood_2019)

#### CycleGAN

Place data in `./datasets/` with image set pairs as `trainA` and `trainB` for training.

#### Flask service

This project was generated with [Angular CLI](https://github.com/angular/angular-cli) version 1.6.0.

##### Development server

Run python server using `python app.py` command

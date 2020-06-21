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

#### Angular/Flask service

This project was generated with [Angular CLI](https://github.com/angular/angular-cli) version 1.6.0.

##### Development server

Run python server using `python server.py` command

##### Development server

Run `ng serve` for a dev server. Navigate to `http://localhost:4200/`. The app will automatically reload if you change any of the source files.

##### Code scaffolding

Run `ng generate component component-name` to generate a new component. You can also use `ng generate directive|pipe|service|class|guard|interface|enum|module`.

##### Build

Run `ng build` to build the project. The build artifacts will be stored in the `dist/` directory. Use the `-prod` flag for a production build.

##### Running unit tests

Run `ng test` to execute the unit tests via [Karma](https://karma-runner.github.io).

##### Running end-to-end tests

Run `ng e2e` to execute the end-to-end tests via [Protractor](http://www.protractortest.org/).


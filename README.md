# ai_soccer
AI to predict football matches

## Technical Info:
- Layes & Neurons
	- 2 Layer first with 19 Neurons and second with 12
	
- Activation function
	- relu
	
- Optimizer
	- Adagrad

- Learning rate	
	- 0.01
	
- steps
	- 5000

- classes
	- 3
	- ['H', 'D', 'A']

## Dataset

Our dataset its composed by 24 variable inputs. This variables are the sum of last recent 20 games. 
- relevant weights: {} = ['home', 'away']
	- {}-wins-home -> 3
	- {}-wins-away -> 4
	- {}-losses-home -> -1.5
	- {}-losses-away -> -1
	- {}-draws-home -> 1
	- draws-away -> 1.5

![Image of api_doc](https://github.com/botclimber/ai_soccer/blob/main/img/dataset_img.png)

## predict

In our predictions we get diff accuracy values, but average always around 50%.

- Italian League:
![Image of api_doc](https://github.com/botclimber/ai_soccer/blob/main/img/ITtest.png)

- Premier League:
![Image of api_doc](https://github.com/botclimber/ai_soccer/blob/main/img/ENtest.png)

- Portuguese League:
![Image of api_doc](https://github.com/botclimber/ai_soccer/blob/main/img/PTtest.png)
![Image of api_doc](https://github.com/botclimber/ai_soccer/blob/main/img/predPT.png)


## How to use

Install dependencies:
```
pip3 install -r requirements.txt 
```
Run Program:
```
python3 predict.py
```

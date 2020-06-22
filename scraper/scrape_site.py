from recipe_scrapers import scrape_me
import pickle
import pandas as pd
from selenium import webdriver
import time
import bs4

import pickle

mega_recipes = pickle.load( open( "sites.p", "rb" ) )
urls = pickle.load( open( "sites2.p", "rb" ) )
mega_recipes = list(set( mega_recipes + urls) )
print(len(mega_recipes), "total recipes")

df = pd.DataFrame(columns=['year', 'title', 'ratings', 'madeit', 'reviews', 'photos', 'submitter_description',
				 'ingredients', 'readyin', 'servings', 'nutrition', 'calories', 'fat', 'carbohydrate', 'protein'])

for i, recipe in enumerate(list(mega_recipes)):
	try:
		s = pd.Series(index=['year', 'title', 'ratings', 'madeit', 'reviews', 'photos', 'submitter_description',
				 'ingredients', 'readyin', 'servings', 'nutrition', 'calories', 'fat', 'carbohydrates', 'protein'])

		# give the url as a string, it can be url from any site listed below
		print(recipe)
		scraper = scrape_me(recipe)
		for div in scraper.soup.find("div", class_="partial recipe-nutrition-section").find_all('div', class_='section-body'):
			s.loc['nutrition'] = div.text

		s.loc['title'] = scraper.title()
		s.loc['readyin'] = scraper.total_time()
		s.loc['servings'] = scraper.yields()
		s.loc['ingredients'] = scraper.ingredients()
		s.loc['instructions'] = scraper.instructions()
		s.loc['photos'] = scraper.image()

		df = df.append(s, ignore_index=True)
		print("Appended", i)
	
	except:
		pass

df.to_csv('allrecipes.csv')
'''
mega_recipes = pickle.load( open( "sites.p", "rb" ) )

#initialize dataframe to store scraped data
df = pd.DataFrame(columns=['year', 'title', 'ratings', 'madeit', 'reviews', 'photos', 'submitter_description',
				 'ingredients', 'readyin', 'servings', 'calories', 'fat', 'carbohydrate', 'protein'])

loc = './chromedriver.exe'
driver = webdriver.Chrome(loc)

#print("MG", mega_recipes)
for recipe_link in list(mega_recipes)[:5]:
	#navigate to recipe
	driver.get(recipe_link)
	
	#initialize series to hold recipe facts
	s = pd.Series(index=['year', 'title', 'ratings', 'madeit', 'reviews', 'photos', 'submitter_description',
			 'ingredients', 'readyin', 'servings', 'calories', 'fat', 'carbohydrate', 'protein'])
	
	try:
		s.loc['title'] = driver.find_elements_by_xpath("//h1[@class='recipe-summary__h1']")[0].text
		print(s.loc['title'])
	except:
		pass
	
	try:
		s.loc['ratings'] = driver.find_elements_by_xpath("//div[@class='rating-stars']")[0].get_attribute('data-ratingstars')
		print(s.loc['ratings'])
	except:
		pass
	
	try:
		s.loc['madeit'] = driver.find_elements_by_xpath("//span[@class='made-it-count ng-binding']")[0].text
		print(s.loc['madeit'])
	except:
		pass
	
	try:
		s.loc['reviews'] = driver.find_elements_by_xpath("//span[@class='review-count']")[0].text.split()[0]
		print(s.loc['reviews'])
	except:
		pass
	
	try:
		s.loc['photos'] = driver.find_elements_by_xpath("//span[@class='picture-count-link']")[0].text.split()[0]
		print(s.loc['photos'])
	except:
		pass
	
	try:
		s.loc['submitter_description'] = driver.find_elements_by_xpath("//div[@class='submitter__description']")[0].text
		print(s.loc['submitter_description'])
	except:
		pass
	
	try:
		s.loc['ingredients'] = [element.text for element in driver.find_elements_by_xpath("//span[@class='recipe-ingred_txt added']")]
		print(s.loc['ingredients'])
	except:
		pass
	
	try:
		s.loc['readyin'] = driver.find_elements_by_xpath("//span[@class='ready-in-time']")[0].text
		print(s.loc['readyin'])
	except:
		pass
	
	try:
		s.loc['servings'] = driver.find_elements_by_xpath("//span[@class='servings-count']//span")[0].text
	except:
		pass
	
	try:
		s.loc['calories'] = driver.find_elements_by_xpath("//span[@class='calorie-count']//span")[0].text
		print(s.loc['calories'])
	except:
		pass
		
	for string in ['fatContent', 'carbohydrateContent', 'proteinContent']:
		try:
			s.loc[re.split(r"[A-Z]", string)[0]] = driver.find_elements_by_xpath("//span[@itemprop='{}']".format(string))[0].text
		except:
			pass
		
	df = df.append(s, ignore_index=True)
	time.sleep(5)

driver.quit()
df.to_csv('allrecipes.csv')
'''
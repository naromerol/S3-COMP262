# COMP262 - NLP AND RECOMMENDER SYSTEMS
# NESTOR ROMERO - 301133331
# ASSIGNMENT 3 - EXERCISE 1

import numpy as np
import pandas as pd
import os
import pprint
from apyori import apriori

### DATA EXPLORATION
# base_path = os.getcwd()
# data = pd.read_json(os.path.join(base_path, 'NestorRomero_COMP262_assignment3','recipes.json'))
# print('DATA HEAD\n', data.head())
# print()
# print('DATA INFO\n', data.info())
# print()
# cuisine_types = data['cuisine'].unique()
# print('CUISINE TYPES\n')
# print('Numtypes: ', len(cuisine_types))
# for ct in cuisine_types:
#     print(ct, end=',')

# recipe_counts = data.groupby(['cuisine']).count()
# pp = pprint.PrettyPrinter()
# pp.pprint(recipe_counts)

class CuisineRecommender: 
    def __init__(self):
        base_path = os.getcwd()
        self.data = pd.read_json(os.path.join(base_path, 'NestorRomero_COMP262_assignment3','recipes.json'))
        self.association_rules = []
        self.rules_map = {}
        # Normalize cuisine removing trailing whitespace and forcing lowercase
        self.data['cuisine'].apply(lambda x : str.lower(x).strip())
        self.cuisine_types = self.data['cuisine'].unique()
        
    def __get_recipes_by_cuisine(self, cuisine):
        
        results = []
        criteria = str.lower(cuisine).strip()
        
        # Validate if criteria is valid
        if criteria in self.cuisine_types:
            filtered_data = self.data[self.data['cuisine'] == criteria]
            results =  filtered_data['ingredients']    
        else:
            results = []
            print(f'>> We don\'t have recommendations for: {criteria}')
            input(f'Type any key to continue')
        
        return results
    
    def __calculate_rules(self, recipes):
        
        if len(recipes) > 0:
            support = 100/len(recipes)
            self.association_rules = apriori(recipes, min_support=support, 
                                             min_confidence=0.5, min_lift=1.0, max_length=None)
            
    def __translate_rules(self):
        
        i = 0
        for item in self.association_rules:
           
            # Not taking single elements into consideration
            if len(item[0]) < 2:
                continue
            
            # Navigate the ordered_statistics subitem
            for rule_set in item[2]:
                
                # Base item could contain a list of elements, order must be ensured
                base_item = list(item[0])
                base_item.sort()
                base_item_key = tuple(base_item)
                rule_items = list(rule_set[0])
                lift = rule_set[3]
                
                # Populate dictionary mapping base_item -> (items, lift)
                if base_item_key not in self.rules_map.keys():
                    self.rules_map[base_item_key] = []
                    
                self.rules_map[base_item_key].append((rule_items,lift))
                
                # print(f'{base_item_key} | {rule_items} | {lift}')
                
        
    def recommend(self, cuisine):
        recipes = self.__get_recipes_by_cuisine(cuisine)    
        self.__calculate_rules(recipes)
        self.__translate_rules()
        
        # Create list of ingredients for cuisine and high lift rules
        ingredients = set([])
        high_lifts = []
        for rule_key in self.rules_map:
            
            # Extract ingredients from rule
            ingredients.update(rule_key) 
            
            # Extract high lift rules
            rule = self.rules_map[rule_key]
            if rule[0][1] > 2.0:
                high_lifts.append((rule_key,(rule[0][0], rule[0][1])))
        
        return ingredients, high_lifts
                


        
# MAIN PROGRAM EXECUTION
reco = CuisineRecommender()
cuisine = ''

while True:
    
    # Clear console for better readability
    os.system('cls')
    
    print('Type a cuisine type for recommendations (i.e Italian)')
    print('Cuisine Types: ')
    print(reco.cuisine_types)
    print()
    cuisine = input('(Type "Exit" to finish) << : ')
    
    # Exit condition
    if str(cuisine).lower().strip() == 'exit':
        break
    
    ingredients, high_lifts = reco.recommend(cuisine)

    if len(ingredients) > 0:
        print(f'TOP INGREDIENTS FOR {str.upper(cuisine)} CUISINE')
        print(ingredients)

    if len(high_lifts) > 0:
        print()
        print()
        print('ITEMS MOST LIKELY TO BE BOUGHT TOGETHER: ')
        for rule in high_lifts:
            print(f'{rule[0]} -> {rule[1][0]} ({rule[1][1]})')
            
    input(f'Type any key to continue')
    
print('END OF PROGRAM')
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 00:19:01 2021
author:Sumit Khedhar

"""

from apyori import apriori
from apyori import load_transactions


class Recommender():
    def __init__(self, inputFile):
        self.AssociationRulesDictionary = {} # holds final output
        self.dataFile = inputFile # input datafile in csv form
        self.association_rules = [] # holds output from Apriori algo

    def computeRules(self):
        """
        Computes all association rules.
        :return:
        """
        with open(self.dataFile ) as fileObj:

            transactions = list(load_transactions(fileObj, delimiter=","))

            # remove empty strings if any
            transactions_filtered = []
            for li in transactions:
                li = list(filter(None, li))
                transactions_filtered.append(li)
            print("this is the first item in the list:",transactions_filtered[0])
            # Following line does all computations
            # lift > 1 shows that there is a positive correlation within the itemset, i.e., items in the
            # itemset, are more likely to be bought together.
            # lift < 1 shows that there is a negative correlation within the itemset, i.e., items in the
            # itemset, are unlikely to be bought together.
            # hence we have set min_lift=1.0 to ignore all rules with lift < 1.0
            self.association_rules = apriori(transactions_filtered, min_support=0.01, min_confidence=0.01, min_lift=1.0,
                                        max_length=None)

    def extractRules(self):

        for item in self.association_rules:
            # first index of the inner list
            # Contains base item and add item

            if len(item[0]) < 2:
                continue

            for k in item[2]:

                baseItemList = list(k[0])
                # if base item set is empty then go to the next record.
                if not baseItemList:
                    continue

                # sort the baseItemList before adding it as a key to the AssociationRules dictionary
                baseItemList.sort()
                baseItemList_key = tuple(baseItemList)

                if baseItemList_key not in self.AssociationRulesDictionary.keys():
                    self.AssociationRulesDictionary[baseItemList_key] = []

                self.AssociationRulesDictionary[baseItemList_key].append((list(k[1]), k[3]))

                # if something goes wrong, then use the following print block to print values
                #print("Base item: ", baseItemList_key)
                #print("Target item: ", list(k[1]))
                #print("Confidence: " + str(k[2]))
                #print("Lift: " + str(k[3]))

        # sort the rules in descending order of lift values.
        for ruleList in self.AssociationRulesDictionary:
            self.AssociationRulesDictionary[ruleList].sort(key=lambda x: x[1], reverse=True)


    def recommend(self, itemList, Num=1):
        """
        itemList is a list of items selected by user
        Num is total recommendations required.
        :param item:
        :return:
        """

        # convert itemList to itemTuple as our dictionary key is a sorted tuple
        itemList.sort()
        itemTuple = tuple(itemList)

        if itemTuple not in self.AssociationRulesDictionary.keys():
            return []

        return self.AssociationRulesDictionary[itemTuple][:Num]

    def studyRules(self):
        """
        This is a template method for computation and rule extraction.
        :return:
        """
        self.computeRules()
        self.extractRules()

    def showDeals(self, itemList, Num=1):
        """
        we are converting the recommendations into deals. The lift value is used to calculate discount percentage
        discount percentage = 10 * lift
        itemList is a list of items selected by user
        Num is total deals required.
        :return:
        """
        recommendations = self.recommend(itemList, Num)

        for item in recommendations:
            print( "If you buy ", item[0], " along with ", itemList, " then you will get ", round((item[1] * 10), 2), \
                   "% discount on total cost!!" )

import os
print(os.getcwd())
Alexa = Recommender(os.path.join(os.getcwd(),"wk09","store_data.csv"))

Alexa.studyRules()

print (Alexa.recommend(['red wine'], 1))

Alexa.showDeals(['red wine'], 2)
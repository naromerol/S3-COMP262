'''
COMP262 - Assignment 1
Nestor Romero - 301133331
Exercise 1
'''
from bs4 import BeautifulSoup
from urllib.request import urlopen

def log_message(file, message):
    file.write(message+'\n')
    print(message)


centennial_ai_website_url = 'https://www.centennialcollege.ca/programs-courses/full-time/artificial-intelligence-online/'
html_source = urlopen(centennial_ai_website_url).read()
soup_parser = BeautifulSoup(html_source, "html.parser")

try:

    with open('nestor_my_future.txt', 'w' ) as output_file:

        log_message(output_file, 'Exercise 1 - Web Scrap')

        log_message(output_file, '\n>> Locate website title')
        title_tag = soup_parser.title
        title = title_tag.getText().strip()
        log_message(output_file, f'Title: {title}')

        #Find parent section containing headings of interest
        career_section = soup_parser.find('div', {'id' : 'tab-3'})
        
        #Locate headings inside section and required contents
        for career_tag in career_section.contents:
            if(career_tag.name == 'h3'):
                print(career_tag.name)
                if( career_tag.getText() == 'Companies Offering Jobs'):
                    #Account for line break
                    companies_offering = career_tag.next_sibling.next_sibling
                    
                if( career_tag.getText() == 'Career Outlook'):
                    #Account for line break
                    possible_careers = career_tag.next_sibling.next_sibling
        
        log_message(output_file, '\n>> Locate companies offering jobs')
        log_message(output_file, 'Companies: ')
        
        companies_list = companies_offering.contents[0].split(', ')
        
        #Add all company names except text: 'and more'
        for c in companies_list[:-1]:
            log_message(output_file, c)
        

        log_message(output_file, '\n>> Locate careers you can pursue')
        log_message(output_file, 'Careers:')
        
        for pctag in possible_careers.contents:
            if(pctag.name == 'li'):
                log_message(output_file, pctag.getText())

except Exception as e:
    print(e)
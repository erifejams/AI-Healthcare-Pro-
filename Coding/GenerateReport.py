###THIS IS TO GENERATE A REPORT FOR THE USER

#pip install seaborn
#pip install fpdf

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from SentimentDisplay import dataFromDatabase
from datetime import datetime, timedelta
from fpdf import FPDF


end_date = pd.Timestamp.today()
start_date = end_date = pd.Timedelta(days = 10 * 365)


def createReport(day, filename = "ConversationReport.pdf"):
    #import database
    dataFromDatabase_Summary = dataFromDatabase.describe()

    #creates a line chart
    sns.replot(data = dataFromDatabase[['sentence', 'smootedSentiment']], kind = 'line', height = 3, aspect = 2.0)
    plt.savefig('graphs/SentimentAnalysisDatabase.png')

    #creates a line chart
    sns.replot(data = dataFromDatabase_Summary[['sentence', 'smootedSentiment']], kind = 'line', height = 3, aspect = 2.0)
    plt.savefig('graphs/SentimentAnalysisDatabaseMean.png')


    #create the pdf report
    #FIRST PAGE
    pdf = FPDF() #uses A4 by default
    pdf.add_page()
    pdf.set_font('Arial', 'B', 24)
    pdf.write(5, f"Conversation Analysis Report")
    pdf.ln(60)
    pdf.set_font('Arial', '', 16)
    pdf.write(4, f'{day}')
    pdf.ln(5)

    #SECOND PAGE
    #what it writes
    pdf.add_page()
    pdf.cell(40, 10, f'Graph Visualisation')
    pdf.image("graphs/SentimentAnalysisDatabase.png", 5, 30, width = 200)
    pdf.image("graphs/SentimentAnalysisDatabaseMean.png", 200/ 2+5, 30, width = 200/2-5)
    pdf.output(filename)

    pdf.write(5, f"thank you for talking with InTa today")


#main
if __name__ == "__main__":
    day = datetime.today()
    createReport(day)
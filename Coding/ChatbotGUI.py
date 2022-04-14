###THIS IS TO CREATE A GRAPHICAL INTERFACE FOR THE CHATBOT
###AND RUN TO HAVE THE COVERSATION WITH THE CHATBOT

from tkinter import *
from DecodeSentences import talk_with_bot, chatbot_name


class ChatApplication:

    def __init__(self):
        self.window = Tk() #gonna be used as a parent as reference in the rest of the code to connect with tk(in tinker)
        self._setup_main_window()


    ###to run the application
    def run(self):
        self.window.mainloop()


    ##this is how the window will look(apperance wise) and contains features such as scrolling, the send messsge botton
    def _setup_main_window(self):
        #title of the window that appears
        self.window.title('Interactive Talker')
        #so the window to be resizeable
        self.window.resizable(width = False, height = False)
        self.window.config(width = 1470, height = 550, bg = 'black')
        
        #the place at the top of the chat where it says what is inserted in the text
        head_label = Label(self.window, bg = 'black', fg = 'white', text = "Hi!!!, my name is Inta, nice to meet you!!! You can just say bye when you want to stop talking to me", pady = 10) #pady to move down
        head_label.place(relwidth = 1)

        #just outline
        line = Label(self.window, width = 450, bg = 'white')
        line.place(relwidth = 1, rely = 0.07, relheight = 0.012)

        #this is the chat area
        self.chatWindow = Text(self.window, bd = 1, bg = 'orange', width = 1200, height = 600, fg = 'black', padx = 0, pady = 0)
        self.chatWindow.place(relheight = 0.8, relwidth = 1, rely = 0.08) #the entire width of the page is used
        #so the users cursor won't be able to click on the box
        self.chatWindow.configure(cursor = "arrow", state = DISABLED)

        
        #firstText =  Label(self.chatWindow, bg = 'white', fg = 'black', text = "InTa: Hi!!!, my name is Inta, nice to meet you!!! You can just say bye when you want to stop talking to me",  pady = 5)
        #firstText.place(relwidth = 1)

        #scrollbar would be able to go down according to the dragging/moving of the text
        scrollbar = Scrollbar(self.chatWindow)
        scrollbar.place(relheight = 1, relx = 0.99)
        scrollbar.configure(command = self.chatWindow.yview)

        #background for the bottom part
        bottom_label = Label(self.window, bg = 'black', height = 80)
        bottom_label.place(relwidth = 1, rely = 0.825)


        #where user can write their messgae
        self.messageWindow = Entry(bottom_label, bg = 'white')
        self.messageWindow.place(relheight = 0.06, relwidth = 0.9, rely = 0.008, relx = 0.011)
        self.messageWindow.focus()
        self.messageWindow.bind("<Return>", self._on_enter_pressed)

        #for user to send message
        #it will get the message eveytime the user pressed enter
        self.Button = Button(bottom_label, text = 'Send', bg = 'grey', activebackground = 'light blue', width = 10, height = 5, font= ('Arial', 20), command = lambda: self._on_enter_pressed(None))
        self.Button.place(relx = 0.77, rely = 0.008, relheight = 0.06, relwidth = 0.22)

       
    #message is inserted in the chat area when enter is pressed
    def _on_enter_pressed(self, event):
        msg = self.messageWindow.get()
        self._insert_message(msg, "You")


    #user can type message in box
    def _insert_message(self, msg, sender):
        #no text is entered
        if not msg:
            return "you didn't write anything"

        if msg == "bye":
            return "you can click finish at the top of the page to go to the next page"
            
        self.messageWindow.delete(0, END)
        userResponse =  f"{sender}: {msg}\n\n"
        self.chatWindow.configure(state = NORMAL)
        ###message is inserted into the chat box
        self.chatWindow.insert(END, userResponse)
        #disable it again
        self.chatWindow.configure(state = DISABLED)

        #gets the chatbot to talk
        intaResponse = f"{chatbot_name}: {talk_with_bot(msg)}\n"
        self.chatWindow.configure(state = NORMAL)
        ###message is inserted into the chat box
        self.chatWindow.insert(END, intaResponse)
        #disable it again
        self.chatWindow.configure(state = DISABLED)

        #Lets you see the last message that is in chat box
        self.chatWindow.see(END)


        
        
    


"""
#create a main menu bar
main_menu = Menu(root)

file_menu = Menu(root)
file_menu.add_command(label = 'New..')
file_menu.add_command(label = 'Save As..')
file_menu.add_command(label = 'Exit..')


main_menu.add_cascade(label = 'File', menu = file_menu)
main_menu.add_command(label = 'Edit')
main_menu.add_command(label = 'Quit')
#adds the menu to the window 
root.config(menu = main_menu )


"""
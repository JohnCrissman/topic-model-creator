import webbrowser

class DisplayNotes():
    def __init__(self, notes):
        string_notes = notes = " ".join(notes)
        notes = string_notes.split(" ")
        notes1 = ""
        
        yellow_words = ["the", "on"] # highlighted yellow
        green_words = ["kid", "dad"] # highlighted green

        for word in notes:
            if(word in yellow_words):
                notes1 = notes1 + '<span class="highlighted-yellow">'+word+'</span>'
            elif(word in green_words):
                notes1 = notes1 + '<span class="highlighted-green">'+word+'</span>'
            else:
                notes1 = notes1 + '<span>'+word+'</span>'
            notes1 = notes1 + " "



        self.notes = notes1
        f = open('helloworld.html','w')

        
        

        message = """<!DOCTYPE>
        <html>
        <head>
        <style>
        .highlighted-green{
            background: #66df66;
        }
        .highlighted-yellow{
            background:yellow;
        }
        </style>
        </head>
        <body><p>""" + self.notes + """</p></body>
        </html>"""


        f.write(message)

        f.close()

        webbrowser.open_new_tab('helloworld.html')

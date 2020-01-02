import webbrowser

f = open('helloworld.html','w')

x = "yo yo yo "
message = """<html>
<head></head>
<body><p>Hello World!""" + x + """</p></body>
</html>"""


f.write(message)

f.close()

webbrowser.open_new_tab('helloworld.html')
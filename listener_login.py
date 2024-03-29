# Run this when new listeners are added to DB before oleg.py

from staff.maintenance import *

listeners = get_listeners()

while True:

    show_listeners(listeners)

    while True:
        print("Listener id to login? Or 'quit'.")
        inp = input()
        try:
            inp = int(inp)
        except ValueError: pass
        if (inp in listeners) or (inp == 'quit'): break

    if inp == 'quit': break

    tdutil = get_tdutil(inp, listeners[inp].phone)
    login_tdlib(tdutil)

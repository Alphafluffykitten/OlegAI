# Flushes selected TDLib database and restores its chat list

from staff.maintenance import *

listeners = get_listeners()


inp = ''
while True:

    show_listeners(listeners)

    while True:
        print("Listener id to flush? Or 'quit'.")
        inp = input()
        try:
            inp = int(inp)
        except ValueError: pass
        if (inp in listeners) or (inp == 'quit'): break

    if inp == 'quit': break

    tdutil = get_tdutil(l_id = inp, phone = listeners[inp].phone)
    destroy_tdlib(tdutil)

    tdutil = get_tdutil(l_id = inp, phone = listeners[inp].phone)
    login_tdlib(tdutil)


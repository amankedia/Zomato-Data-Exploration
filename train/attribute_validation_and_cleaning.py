def onlineOrderMapping(online_order):
    return online_order.map({"Yes": 1, "No":0})

def tableBookingMapping(book_table):
    return book_table.map({"Yes": 1, "No":0})
import gspread

gc = gspread.service_account(filename="./tools/creds.json")

# Open a sheet from a spreadsheet in one go
spreadsheet = gc.open("Backtest result strategy")
second_sheet = spreadsheet.get_worksheet(1)

if second_sheet.title.lower() == "test":
    # Update a range of cells using the top left corner address
    second_sheet.update([[1, 2], [3, 4]], "A1")
    second_sheet.update_acell(
        "B42", "it's down there somewhere, let me take another look."
    )
    second_sheet.format("A1:B1", {"textFormat": {"bold": True}})
else:
    print("List with name 'test' is not found")

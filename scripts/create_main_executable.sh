# Ensure you have set up the virtual environment before running this.
# Also make sure to do: pip install pyinstaller

# Have to add 'externals' as data as we call a script from that folder
pyinstaller --add-data "externals";"externals" \
            --noconfirm --onefile --windowed main.py
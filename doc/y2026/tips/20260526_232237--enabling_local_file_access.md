
## how to enable chromium browser open an local HTML file?

You can open a local HTML file in Chromium by simply dragging and dropping the file into an open browser window, or by using the keyboard shortcut Ctrl + O (Cmd + O on Mac) to browse and select it.

If you need the browser to have permission to access other local files (like loading local JSON or images via JavaScript), you must launch it with a specific security flag.

### Quick Ways to Open a Local File
1. Drag and Drop: Grab your .html file from your folder and drop it into a Chromium tab.
1. File Open Shortcut: Press Ctrl + O (Windows/Linux) or Cmd + O (Mac) inside Chromium to open the file picker.
1. Address Bar: Type the direct path into the address bar using the file:// protocol (e.g., file:///C:/Users/Name/Desktop/index.html on Windows or file:///home/user/index.html on Linux/Mac).

### Enabling Local File Access (Developer Mode)
Due to security restrictions like **CORS**, some local HTML files (especially those using scripts to load other files) may not work correctly by default. You can enable full access by launching Chromium with the <code>--allow-file-access-from-files</code> flag.

#### For Windows:
1. Close all running instances of Chromium.
1. Right-click your Chromium shortcut and select Properties.
1. In the Target field, add a space at the end and then paste: <code>--allow-file-access-from-files</code>.
1. Click OK and launch the browser using that shortcut.

#### For Mac:
1. Close Chromium completely.
1. Open the Terminal app.
1. Type the following command and press Enter: <code>open /Applications/Chromium.app --args --allow-file-access-from-files</code>

#### For Linux (Ubuntu/Debian):
1. Open your terminal.
1. Run Chromium with the flag:chromium-browser <code>--allow-file-access-from-files</code>

Warning: Using the <code>--allow-file-access-from-files</code> flag lowers browser security. Only use it for local development and avoid browsing the general web while it is active. For ongoing development, consider using a simple local web server instead.



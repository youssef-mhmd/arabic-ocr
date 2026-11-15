# Installing arabic-ocr

arabic-ocr reads Arabic, Persian and Urdu text from images and PDFs. It runs on
your own computer, offline: nothing is uploaded anywhere. The installer contains
everything it needs, so you do **not** need Python.

Download the installer for your system from the
[Releases page](https://github.com/youssef-mhmd/arabic-ocr/releases/latest).
Each installer is about 100–150 MB.

| System | Download | Requirements |
|---|---|---|
| Windows | `arabic-ocr-<version>-windows-x64-setup.exe` | Windows 10 or 11, 64-bit |
| macOS | `arabic-ocr-<version>-macos-arm64.dmg` | Apple Silicon Mac (M1 or newer), macOS 14 Sonoma or newer |
| Linux | `arabic-ocr_<version>_amd64.deb` | Ubuntu 22.04 or newer (or Debian 12+, Linux Mint 21+), 64-bit Intel/AMD |

At least 4 GB of RAM is needed (8 GB recommended). No graphics card is needed.

Each install gives you two programs:
- **Arabic OCR**, the desktop app (`arabic-ocr-gui`).
- **`arabic-ocr`**, the command-line tool ([CLI reference](cli.md)).

To check that a download is complete and unchanged, compare its SHA-256 with
`SHA256SUMS.txt` on the release page:
`certutil -hashfile <file> SHA256` (Windows), `shasum -a 256 <file>` (macOS),
`sha256sum <file>` (Linux).

---

## Windows

1. Run `arabic-ocr-<version>-windows-x64-setup.exe`.
2. **"Windows protected your PC" (SmartScreen):** the installer is not code-signed
   yet, so Windows warns about it. Click **More info**, then **Run anyway**.
3. Follow the steps. By default it installs for your user only, so no
   administrator rights are needed. Options:
   - *Create a desktop shortcut*
   - *Add the arabic-ocr command-line tool to PATH*: tick this if you want to
     type `arabic-ocr` in a terminal. Open a **new** terminal afterwards.
4. Start **Arabic OCR** from the Start menu.

The default install folder is `%LOCALAPPDATA%\Programs\Arabic OCR`. To uninstall,
go to **Settings > Apps > Installed apps > Arabic OCR > Uninstall**.

Silent install, e.g. for IT departments:
`arabic-ocr-<version>-windows-x64-setup.exe /VERYSILENT /TASKS=addtopath`

## macOS (Apple Silicon)

1. Open `arabic-ocr-<version>-macos-arm64.dmg`.
2. Drag **Arabic OCR** onto the **Applications** folder.
3. **First start:** the app is not notarized by Apple yet, so macOS blocks it the
   first time ("Apple could not verify…"). Allow it once:
   - Open **Applications**, then **Control-click** (or right-click) **Arabic OCR**
     and choose **Open**, then **Open** again. On macOS 15 and later, if that
     doesn't offer Open: try to open the app once, then go to **System Settings >
     Privacy & Security**, scroll down and click **Open Anyway**.
   - Or, in Terminal: `xattr -dr com.apple.quarantine "/Applications/Arabic OCR.app"`

After that it opens normally. You can also use **Open With > Arabic OCR** on
images and PDFs in Finder.

The command-line tool is inside the app. To use it in Terminal, link it once:

```bash
sudo ln -sf "/Applications/Arabic OCR.app/Contents/MacOS/arabic-ocr" /usr/local/bin/arabic-ocr
```

To uninstall, drag **Arabic OCR** from Applications to the Bin.

Intel Macs are not supported: the libraries the app is built on no longer
publish Intel Mac versions.

## Linux (Ubuntu, Debian, Mint)

```bash
sudo apt install ./arabic-ocr_<version>_amd64.deb
```

Use `apt` (not `dpkg -i`) so the few system libraries the app needs are
installed too. Then start **Arabic OCR** from the applications menu, or run
`arabic-ocr-gui`. The command-line tool is `arabic-ocr`.

To uninstall: `sudo apt remove arabic-ocr`.

The app is installed in `/opt/arabic-ocr`. On Wayland desktops it runs through
XWayland.

---

## Using the desktop app

- **Open** images or PDFs (PNG, JPEG, TIFF with any number of pages, BMP, WebP,
  PDF), **Open folder** to read every file in a folder, or **drag and drop**
  files onto the window.
- **Paste image** (Ctrl+V; ⌘V on macOS) reads a screenshot from the clipboard.
- Files are read one after another in the background. The list on the left shows
  each file's progress; **Cancel** stops the queue.
- Click a file to see it: the page with a box around every line it found, and
  the text next to it. Click a box to jump to its text, or put the cursor in a
  line of text to see its box. Use the arrows for the pages of a PDF or TIFF.
- **The text can be edited.** Your corrections are kept and used by **Copy text**
  and every export.
- **Export…** saves the selected file as plain text, JSON (with boxes and
  confidence), a **searchable PDF** (the original pages with invisible, selectable
  text), hOCR or ALTO XML. **Export all…** saves every finished file to the
  output folder in the format chosen in Settings.
- **Settings:** keep or remove diacritics (tashkeel) and tatweel (ـ); digits as
  printed, Western (0–9) or Arabic-Indic (٠–٩); output folder; interface
  language. These options change the shown text and exports at once, with no
  need to read the files again.
- **العربية / English** switches the interface language. The whole window is
  mirrored for Arabic.
- **Try a sample** reads a short example text.

## Checking an installation

`arabic-ocr-gui --self-test` reads the built-in sample, exports every format and
prints PASS/FAIL for each step. On Windows the app has no console, so give it a
file for the report: `arabic-ocr-gui.exe --self-test report.txt`.

## Problems?

- **Nothing happens / an error about models:** reinstall. The two model files
  (about 27 MB) are part of the installer.
- **Linux: "could not load the Qt platform plugin xcb":** install the library it
  names, e.g. `sudo apt install libxcb-cursor0`.
- Report other problems on the
  [issue tracker](https://github.com/youssef-mhmd/arabic-ocr/issues).

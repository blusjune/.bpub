" 20230701_103934
" 20261026_232156
set nocompatible
set nu ai cin
set ignorecase


"Please
"Open this .vimrc file, and
"execute the ex command :PlugInstall
call plug#begin()
  Plug 'aklt/plantuml-syntax'             " syntax hl for puml
  Plug 'tyru/open-browser.vim'            " hooks for opeing browser
  Plug 'weirongxu/plantuml-previewer.vim' " previewer

  " tabular plug-in (recommended for table-sort feature)
  Plug 'godlygeek/tabular'

  " markdown plug-in core
  Plug 'plasticboy/vim-markdown'
  Plug 'iamcco/markdown-preview.nvim', { 'do': { -> mkdp#util#install() } }

  " CSV file plug-in
  Plug 'mechatroner/rainbow_csv'
  "Plug 'chrisbra/csv.vim'

  " JSON plug-in
  Plug 'elzr/vim-json'
call plug#end()


let g:vim_markdown_fenced_languages = ['python=python', 'bash=sh', 'json=json']



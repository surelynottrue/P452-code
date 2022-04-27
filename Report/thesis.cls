\ProvidesClass{thesis}
\NeedsTeXFormat{LaTeX2e}
\LoadClass[a4paper,11pt]{book}

\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{titlesec}
\usepackage{lipsum}
\usepackage{contour}
\usepackage{ulem}
% \usepackage[bitstream-charter]{mathdesign}
% \usepackage[T1]{fontenc}
\usepackage{microtype}
\usepackage[colorlinks=true, linkcolor=blue, citecolor=blue]{hyperref}
\usepackage[style=science,backend=biber]{biblatex}
\usepackage[hybrid]{markdown}
\usepackage{etoolbox}
\AtBeginEnvironment{quotation}{\itshape}
\addbibresource{ref.bib}
\edef\restoreparindent{\parindent=\the\parindent\relax}
\usepackage{parskip}
\restoreparindent
% \setlength{\parskip}{0.5em}

\renewcommand{\ULdepth}{2pt}
\contourlength{0.8pt}

\newcommand{\myuline}[1]{%
  \uline{\phantom{#1}}%
  \llap{\contour{white}{#1}}%
}

\titleformat{\chapter}[display]{\bfseries\centering}{\vspace{15em} \normalsize CHAPTER \thechapter}{0.1em}{\huge}
\titlespacing\chapter{0pt}{0pt plus 0pt minus 0pt}{10pt plus 0pt minus 0pt}
\titleformat{\section}{\normalfont\large\bfseries}{\arabic{section}.}{0.5em}{\MakeUppercase}[]
\titleformat{\subsubsection}{\large\itshape}{}{0.0em}{\myuline}[]
\titleformat{\subsection}{\large\bfseries}{\arabic{section}.\arabic{subsection}.}{0.4em}{}[]

% tightlist 
\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
% enumerate style changes
\renewcommand{\labelenumi}{\arabic{enumi}) }
\setlength\columnsep{2em}

% cite and url fontstyle changes
\renewcommand{\UrlFont}{\normalfont}
\renewrobustcmd*{\mkbibemph}{}
\protected\long\def\blx@imc@mkbibemph#1{#1}

\DeclareCiteCommand{\cite}[\mkbibsuperscript]
  {
   \usebibmacro{cite:init}%
   \let\multicitedelim=\supercitedelim
   \iffieldundef{prenote}
     {}
     {\BibliographyWarning{Ignoring prenote argument}}%
   \iffieldundef{postnote}
     {}
     {\BibliographyWarning{Ignoring postnote argument}}}
  {\usebibmacro{citeindex}%
   \usebibmacro{cite:comp}}
  {}
  {\usebibmacro{cite:dump}%
   }
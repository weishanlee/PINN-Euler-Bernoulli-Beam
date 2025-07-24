% Custom version of sn-jnl.cls with modified corresponding author text
% This file extends the original sn-jnl.cls

% Load the original class
\input{sn-jnl.cls}

% Redefine the \@maketitle command to remove the (s) from Corresponding author text
\renewcommand{\@maketitle}{\newpage\null%
    \if@remarkboxon\vbox to 0pt{\vspace*{-78pt}\hspace*{-18pt}\FMremark}\else\vskip21pt\fi%%\par%
    \hsize\textwidth\parindent0pt%%%\vskip7pt%
    %% Aritle Type
    {\hbox to \textwidth{{\Artcatfont\ArtType\hfill}\par}}
    %% Aritle Title
    \ifx\@title\empty\else%
        \removelastskip\vskip20pt\nointerlineskip%
        {\Titlefont\@title\par}
        %\addcontentsline{toc}{chapter}{\@title}% for bookmarks
    \fi%
    %% Aritle SubTitle
    \ifx\@subtitle\empty\else%
        \vskip9pt%
        {{\SubTitlefont\@subtitle\par}}
    \fi%
    %% Aritle Authors, Address and Correspondings
    \ifnum\aucount>0
        \global\punctcount\aucount%
        \vskip20pt%
        \artauthors\par%%     authors and emails
        {\vskip7pt\addressfont\auaddress\par%%      corresponding adress
	 \removelastskip\vskip24pt%
	\ifnum\emailcnt>0\relax%
           \ifx\corrauthemail\@empty\else{\ifnum\aucount>1*\fi}%
	   Corresponding author. E-mail: \corrauthemail\par\fi%  % Changed here - removed (s)
	   \ifx\authemail\@empty\else Contributing authors:\ \authemail\fi%
        \fi%
        \ifequalcont{\par$^{\dagger}$\@equalconttext\par}\fi%
	 \removelastskip\vskip24pt%
        \ifpresentaddress{\par\@presentaddresstext\par}\fi%
	}
     \fi%
     {\printabstract\par}%
     {\printkeywords\par}%
     \ifx\@pacs\empty\else%
       \loop\ifnum\PacsCount>0%
          \csname\romannumeral\PacsTmpCnt StorePacsTxt\endcsname\par%
          \StepDownCounter{\PacsCount}%
          \StepUpCounter{\PacsTmpCnt}%
       \repeat%
    \fi%
    %%{\printhistory\par}%
    %%{\ifx\@motto\empty\else\@motto\fi}%
    \removelastskip\vskip36pt\vskip0pt}%
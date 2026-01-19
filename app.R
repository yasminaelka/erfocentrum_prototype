#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    https://shiny.posit.co/
#

#
# app.R — Erfocentrum prototype + AI-navigatie prompt (fixed nav)
#

library(shiny)
library(readr)
library(dplyr)
library(bslib)
library(stringr)

`%||%` <- function(x, y) if (is.null(x)) y else x

# =========================
# DATA: main content
# =========================
content_raw <- read_csv("data/dsp_dataset_erfocentrum_v1.csv", show_col_types = FALSE)

content_ui <- content_raw %>%
  transmute(
    doc_id,
    title = name,
    category,
    url,
    short = short_text_simple,
    long  = long_text_complex,
    next_step,
    disclaimer,
    aliases
  ) %>%
  distinct(doc_id, .keep_all = TRUE) %>%
  mutate(
    category = ifelse(is.na(category) | category == "", "Overig", category),
    short = ifelse(is.na(short) | short == "", "", short),
    long  = ifelse(is.na(long)  | long  == "",  "", long),
    next_step = ifelse(is.na(next_step) | next_step == "", "", next_step),
    disclaimer = ifelse(is.na(disclaimer) | disclaimer == "", "", disclaimer),
    aliases = ifelse(is.na(aliases) | aliases == "", "", aliases),
    url = ifelse(is.na(url) | url == "", "", url)
  )

cats <- sort(unique(content_ui$category))
cats <- c("Alles", cats)

# =========================
# DATA: extra datasets
# =========================
themes <- read_csv("data/themes.csv", show_col_types = FALSE)
home_featured <- read_csv("data/home_featured.csv", show_col_types = FALSE)
hl_blocks <- read_csv("data/hl_support_blocks.csv", show_col_types = FALSE)
glossary <- read_csv("data/glossary.csv", show_col_types = FALSE)
syn <- read_csv("data/synonyms.csv", show_col_types = FALSE)

# Normalize synonyms
syn <- syn %>%
  mutate(
    term = str_trim(tolower(term)),
    synonyms = str_trim(tolower(synonyms))
  )

# =========================
# Helpers: concepts + synonyms + spelling + glossary hits
# =========================
expand_with_synonyms <- function(query, syn_df) {
  q <- str_trim(tolower(query))
  if (!nzchar(q)) return(character(0))
  
  hit <- syn_df[syn_df$term == q, , drop = FALSE]
  if (nrow(hit) == 0) return(q)
  
  extra <- unlist(strsplit(hit$synonyms[1], ","))
  extra <- str_trim(tolower(extra))
  unique(c(q, extra))
}

expand_with_concepts <- function(query, syn_df, themes_df) {
  base <- expand_with_synonyms(query, syn_df)
  
  q <- str_trim(tolower(query))
  theme_terms <- character(0)
  if (!is.null(themes_df) && nrow(themes_df) > 0) {
    labs <- tolower(themes_df$label)
    if (any(str_detect(labs, fixed(q)) | str_detect(q, fixed(labs)))) {
      theme_terms <- labs
    }
  }
  
  unique(c(base, theme_terms))
}

best_spelling_match <- function(q, vocab, max_dist = 2) {
  q <- trimws(tolower(q))
  if (!nzchar(q)) return(NA_character_)
  
  v <- unique(trimws(tolower(vocab)))
  v <- v[nzchar(v)]
  if (length(v) == 0) return(NA_character_)
  
  d <- adist(q, v)
  i <- which.min(d)
  if (length(i) == 0) return(NA_character_)
  if (d[i] <= max_dist) return(v[i])
  NA_character_
}

extract_glossary_hits <- function(text, glossary_df, max_n = 6) {
  if (is.na(text) || !nzchar(text)) return(glossary_df[0, , drop = FALSE])
  t_low <- tolower(text)
  
  glossary_df %>%
    mutate(term_low = tolower(term)) %>%
    filter(str_detect(t_low, fixed(term_low))) %>%
    select(term, uitleg) %>%
    slice_head(n = max_n)
}

# =========================
# NEW: "AI" Navigator helpers (heuristic bot)
# =========================
tokenize_nl <- function(text) {
  x <- tolower(text %||% "")
  x <- str_replace_all(x, "[^a-z0-9à-ÿ\\s-]", " ")
  x <- str_replace_all(x, "\\s+", " ")
  x <- str_trim(x)
  if (!nzchar(x)) return(character(0))
  toks <- unlist(str_split(x, " "))
  toks <- toks[nchar(toks) >= 3]
  toks <- toks[!toks %in% c("een","het","dat","wat","hoe","waar","waarom","met","naar","voor","van","mijn","jouw","jij","ik","ook","niet","wel","kun","kan","wil","wilt","ben","zijn","was","dit","deze","die","dus","als","bij","nog")]
  unique(toks)
}

score_docs_for_prompt <- function(prompt, content_df, themes_df, syn_df, top_n = 5) {
  p <- str_trim(prompt %||% "")
  if (!nzchar(p)) return(content_df[0, , drop = FALSE])
  
  toks <- tokenize_nl(p)
  
  syn_add <- character(0)
  if (!is.null(syn_df) && nrow(syn_df) > 0 && length(toks) > 0) {
    for (t in toks) {
      hit <- syn_df[syn_df$term == t, , drop = FALSE]
      if (nrow(hit) > 0) {
        extra <- unlist(strsplit(hit$synonyms[1], ","))
        extra <- str_trim(tolower(extra))
        syn_add <- c(syn_add, extra)
      }
    }
  }
  toks <- unique(c(toks, syn_add))
  toks <- toks[nzchar(toks)]
  
  theme_hits <- character(0)
  if (!is.null(themes_df) && nrow(themes_df) > 0) {
    labs <- themes_df$label
    labs_low <- tolower(labs)
    for (i in seq_along(labs_low)) {
      if (nzchar(labs_low[i]) && str_detect(tolower(p), fixed(labs_low[i]))) {
        theme_hits <- c(theme_hits, labs[i])
      }
    }
  }
  theme_hits <- unique(theme_hits)
  
  df <- content_df %>%
    mutate(
      t_title = tolower(title %||% ""),
      t_short = tolower(short %||% ""),
      t_long  = tolower(long %||% ""),
      t_alias = tolower(aliases %||% ""),
      t_cat   = tolower(category %||% ""),
      t_all   = paste(t_title, t_short, t_alias, t_cat, sep = " | ")
    )
  
  score <- rep(0, nrow(df))
  for (t in toks) {
    score <- score +
      3L * str_detect(df$t_title, fixed(t)) +
      2L * str_detect(df$t_short, fixed(t)) +
      2L * str_detect(df$t_alias, fixed(t)) +
      1L * str_detect(df$t_cat, fixed(t))
  }
  
  if (length(theme_hits) > 0) {
    for (th in theme_hits) {
      th_low <- tolower(th)
      score <- score + 2L * str_detect(df$t_all, fixed(th_low))
    }
  }
  
  df %>%
    mutate(score = score) %>%
    filter(score > 0) %>%
    arrange(desc(score)) %>%
    select(doc_id, title, category, url, short, long, next_step, disclaimer, aliases, score) %>%
    slice_head(n = top_n)
}

next_question_for_prompt <- function(prompt) {
  p <- tolower(prompt %||% "")
  if (!nzchar(p)) return("Kun je in één zin vertellen wat je wilt weten? Bijvoorbeeld: kinderwens, dragerschap, DNA-onderzoek, of een uitslag.")
  if (str_detect(p, "zwanger|kinderwens|ivf|pgd|pgt")) return("Gaat het om kinderwens (vooraf testen) of om een uitslag die je al hebt gekregen?")
  if (str_detect(p, "uitslag|resultaat|test|dna")) return("Weet je om welk type DNA-onderzoek of uitslag het gaat (bijv. dragerschap, diagnostiek, prenataal)?")
  if (str_detect(p, "familie|erfelijk|ouders|broer|zus")) return("Gaat het om een erfelijke aandoening in de familie, of wil je weten of jij drager bent?")
  "Wil je vooral uitleg (wat betekent het), of vooral wat je nu het beste kunt doen (volgende stap)?"
}

# =========================
# UI
# =========================
ui <- page_navbar(
  id = "topnav",  # <<<<<< IMPORTANT for updateNavbarPage
  title = "Erfocentrum (Prototype)",
  theme = bs_theme(version = 5),
  header = tags$head(
    tags$link(rel = "stylesheet", type = "text/css", href = "styles.css")
  ),
  
  nav_menu("DNA-onderzoek", nav_panel("Zoeken", value = "search")),
  nav_menu("Ziektes (en dan?)", nav_panel("Zoeken", value = "search")),
  nav_menu("Kinderwens", nav_panel("Zoeken", value = "search")),
  nav_menu("Familie of niet", nav_panel("Zoeken", value = "search")),
  
  nav_panel(
    "Voorpagina",
    value = "home",
    div(
      class = "page-wrap",
      card(
        class = "hero-card",
        h2("Erfelijkheid gaat over iedereen. Ook over jou."),
        p("Prototype met verbeterde zoekfunctie (concepten, synoniemen, spelling, suggesties) en weergaveniveaus Basis/Standaard/Uitgebreid."),
        actionButton("go_search", "Ga naar Zoeken", class = "btn-primary")
      ),
      br(),
      layout_columns(
        col_widths = c(4, 4, 4),
        uiOutput("home_col1"),
        uiOutput("home_col2"),
        uiOutput("home_col3")
      )
    )
  ),
  
  nav_panel(
    "Zoeken",
    value = "search",
    layout_sidebar(
      sidebar = sidebar(
        radioButtons(
          "hl",
          "Weergaveniveau",
          choices = c("Basis" = "low", "Standaard" = "mid", "Uitgebreid" = "high"),
          selected = "mid",
          inline = TRUE
        ),
        
        selectizeInput(
          "q",
          "Zoekterm",
          choices = NULL,
          multiple = FALSE,
          options = list(
            placeholder = "Bijv. ‘spierziekte’, ‘drager’, ‘DNA-onderzoek’",
            create = TRUE
          )
        ),
        
        selectInput(
          "situation",
          "Zoek op situatie",
          choices = c(
            "Kies een situatie…" = "",
            "Ik wil zwanger worden / kinderwens" = "kinderwens",
            "Er is een erfelijke ziekte in mijn familie" = "ziekte in familie",
            "Ik wil weten of ik drager ben" = "drager",
            "Ik begrijp de uitslag van DNA-onderzoek niet" = "dna uitslag",
            "Ik wil weten wat DNA-onderzoek is" = "dna onderzoek"
          )
        ),
        
        selectInput("cat", "Categorie", choices = cats),
        checkboxInput("use_aliases", "Zoek ook in aliassen", value = TRUE),
        
        uiOutput("syn_hint"),
        uiOutput("spell_hint"),
        
        tags$hr(),
        
        div(class = "meta", "Veelgezocht (klik):"),
        fluidRow(
          column(6, actionButton("quick_drager", "Drager", class = "btn-sm btn-outline-secondary")),
          column(6, actionButton("quick_dna", "DNA-onderzoek", class = "btn-sm btn-outline-secondary"))
        ),
        br(),
        fluidRow(
          column(6, actionButton("quick_kinderwens", "Kinderwens", class = "btn-sm btn-outline-secondary")),
          column(6, actionButton("quick_familie", "Familie", class = "btn-sm btn-outline-secondary"))
        ),
        
        tags$hr(),
        
        actionButton("browse_all", "Ik weet niet welk woord ik moet gebruiken", class = "btn-outline-primary"),
        uiOutput("theme_chips"),
        
        tags$hr(),
        actionButton("clear", "Reset", class = "btn-outline-secondary")
      ),
      
      card(
        h3("Resultaten"),
        uiOutput("results_count"),
        uiOutput("results_list")
      )
    )
  ),
  
  nav_panel(
    "Detail",
    value = "detail",
    card(
      h2(textOutput("detail_title")),
      p(class = "meta", textOutput("detail_meta")),
      uiOutput("detail_layer"),
      tags$hr(),
      p(textOutput("detail_short")),
      tags$details(
        tags$summary("Lees volledige tekst"),
        p(textOutput("detail_long"))
      ),
      tags$hr(),
      h4("Volgende stap"),
      p(textOutput("detail_next")),
      h4("Disclaimer"),
      p(textOutput("detail_disclaimer")),
      br(),
      actionButton("back_to_search", "← Terug naar resultaten", class = "btn-outline-secondary")
    )
  )
)

# =========================
# SERVER
# =========================
server <- function(input, output, session) {
  selected_id <- reactiveVal(NULL)
  
  bot_state <- reactiveValues(
    history = list(),
    last_prompt = "",
    recs = content_ui[0, , drop = FALSE]
  )
  
  render_home_slot <- function(slot_n) {
    slot <- home_featured %>% filter(slot == slot_n) %>% slice(1)
    if (nrow(slot) == 0) return(NULL)
    
    a <- content_ui %>% filter(doc_id == slot$doc_id[1]) %>% slice(1)
    if (nrow(a) == 0) return(NULL)
    
    card(
      class = "teaser-card",
      div(class = "teaser-kicker", slot$label[1]),
      tags$strong(a$title),
      if (nzchar(a$short)) p(a$short),
      actionButton(paste0("open_", a$doc_id), "Lees meer", class = "btn-sm btn-primary")
    )
  }
  
  output$home_col1 <- renderUI(render_home_slot(1))
  output$home_col2 <- renderUI(render_home_slot(2))
  output$home_col3 <- renderUI(render_home_slot(3))
  
  observeEvent(input$go_search, {
    updateNavbarPage(session, id = "topnav", selected = "search")
  })
  
  observe({
    vocab <- unique(c(content_ui$title, content_ui$aliases, content_ui$category))
    vocab <- vocab[!is.na(vocab) & nzchar(vocab)]
    vocab <- sort(unique(trimws(vocab)))
    updateSelectizeInput(session, "q", choices = vocab, server = TRUE)
  })
  
  observeEvent(input$situation, {
    if (!nzchar(input$situation)) return()
    updateSelectizeInput(session, "q", selected = input$situation)
    updateRadioButtons(session, "hl", selected = "low")
  }, ignoreInit = TRUE)
  
  observeEvent(input$quick_drager, {
    updateSelectizeInput(session, "q", selected = "drager")
    updateRadioButtons(session, "hl", selected = "low")
  })
  observeEvent(input$quick_dna, {
    updateSelectizeInput(session, "q", selected = "dna onderzoek")
    updateRadioButtons(session, "hl", selected = "low")
  })
  observeEvent(input$quick_kinderwens, {
    updateSelectizeInput(session, "q", selected = "kinderwens")
    updateRadioButtons(session, "hl", selected = "low")
  })
  observeEvent(input$quick_familie, {
    updateSelectizeInput(session, "q", selected = "familie")
    updateRadioButtons(session, "hl", selected = "low")
  })
  
  # ---- AI modal ----
  bot_ui <- function() {
    div(
      tags$p("Beschrijf je situatie of vraag in gewone taal. Vervolgens zal de navigatiebot je helpen navigeren naar de juiste informatie."),
      textAreaInput(
        "ai_prompt",
        "Jouw vraag / probleem",
        placeholder = "Bijv. 'Mijn zus heeft een erfelijke spierziekte, wat betekent dit voor mij?'",
        width = "100%",
        height = "120px"
      ),
      fluidRow(
        column(6, actionButton("ai_send", "Start AI-navigatie", class = "btn-primary")),
        column(6, actionButton("ai_reset", "Reset gesprek", class = "btn-outline-secondary"))
      ),
      tags$hr(),
      uiOutput("ai_chat"),
      tags$hr(),
      uiOutput("ai_recs")
    )
  }
  
  show_ai_modal <- function() {
    showModal(modalDialog(
      title = "AI-navigatie (prototype)",
      size = "l",
      easyClose = TRUE,
      footer = modalButton("Sluiten"),
      bot_ui()
    ))
  }
  
  observeEvent(input$browse_all, {
    updateRadioButtons(session, "hl", selected = "low")
    show_ai_modal()
  }, ignoreInit = TRUE)
  
  output$ai_chat <- renderUI({
    h <- bot_state$history
    if (length(h) == 0) {
      return(div(class = "meta", "Nog geen gesprek. Typ je vraag en klik op ‘Start AI-navigatie’."))
    }
    tagList(lapply(seq_along(h), function(i) {
      msg <- h[[i]]
      role <- msg$role
      cls <- if (role == "user") "border rounded p-2 mb-2" else "border rounded p-2 mb-2 bg-light"
      tags$div(
        class = cls,
        tags$div(style = "font-weight:600;", if (role == "user") "Jij" else "Navigator"),
        tags$div(msg$text)
      )
    }))
  })
  
  output$ai_recs <- renderUI({
    recs <- bot_state$recs
    if (is.null(recs) || nrow(recs) == 0) {
      return(div(class = "meta", "Nog geen suggesties."))
    }
    
    tagList(
      tags$h4("Aanbevolen informatie"),
      tagList(lapply(seq_len(nrow(recs)), function(i) {
        r <- recs[i, ]
        card(
          class = "result-card",
          tags$strong(r$title),
          div(class = "meta", paste0(r$category, ifelse(nzchar(r$url), paste0(" • ", r$url), ""))),
          if (nzchar(r$short)) p(r$short),
          actionButton(paste0("ai_open_", r$doc_id), "Open", class = "btn-sm btn-primary"),
          actionButton(paste0("ai_use_query_", r$doc_id), "Zoek hiermee", class = "btn-sm btn-outline-secondary")
        )
      }))
    )
  })
  
  observeEvent(input$ai_send, {
    p <- str_trim(input$ai_prompt %||% "")
    if (!nzchar(p)) return()
    
    bot_state$last_prompt <- p
    bot_state$history <- append(bot_state$history, list(list(role = "user", text = p)))
    
    recs <- score_docs_for_prompt(p, content_ui, themes, syn, top_n = 5)
    bot_state$recs <- recs
    
    qn <- next_question_for_prompt(p)
    bot_state$history <- append(
      bot_state$history,
      list(list(role = "bot", text = paste0(
        "Ik heb je tekst gekoppeld aan zoekwoorden/onderwerpen en daaruit suggesties gemaakt.\n\nVervolgvraag: ",
        qn
      )))
    )
  }, ignoreInit = TRUE)
  
  observeEvent(input$ai_reset, {
    bot_state$history <- list()
    bot_state$last_prompt <- ""
    bot_state$recs <- content_ui[0, , drop = FALSE]
    updateTextAreaInput(session, "ai_prompt", value = "")
  }, ignoreInit = TRUE)
  
  observe({
    recs <- bot_state$recs
    if (is.null(recs) || nrow(recs) == 0) return()
    lapply(recs$doc_id, function(id) {
      observeEvent(input[[paste0("ai_open_", id)]], {
        selected_id(id)
        removeModal()
        updateNavbarPage(session, id = "topnav", selected = "detail")
      }, ignoreInit = TRUE)
      
      observeEvent(input[[paste0("ai_use_query_", id)]], {
        row <- content_ui %>% filter(doc_id == id) %>% slice(1)
        updateSelectizeInput(session, "q", selected = row$title[1])
        updateSelectInput(session, "cat", selected = "Alles")
        updateRadioButtons(session, "hl", selected = "low")
        removeModal()
        updateNavbarPage(session, id = "topnav", selected = "search")
      }, ignoreInit = TRUE)
    })
  })
  
  # Theme chips
  output$theme_chips <- renderUI({
    if (is.null(themes) || nrow(themes) == 0) return(NULL)
    chips <- lapply(seq_len(nrow(themes)), function(i) {
      id <- paste0("theme_", themes$theme_id[i])
      actionButton(id, themes$label[i], class = "btn-sm btn-outline-secondary", style = "margin:2px;")
    })
    tagList(div(class = "meta", "Onderwerpen:"), tagList(chips))
  })
  
  observe({
    if (is.null(themes) || nrow(themes) == 0) return()
    lapply(seq_len(nrow(themes)), function(i) {
      id <- paste0("theme_", themes$theme_id[i])
      observeEvent(input[[id]], {
        updateSelectizeInput(session, "q", selected = themes$label[i])
        updateSelectInput(session, "cat", selected = "Alles")
        updateRadioButtons(session, "hl", selected = "low")
      }, ignoreInit = TRUE)
    })
  })
  
  observeEvent(input$clear, {
    updateSelectizeInput(session, "q", selected = "", choices = NULL)
    updateSelectInput(session, "cat", selected = "Alles")
    updateCheckboxInput(session, "use_aliases", value = TRUE)
    updateSelectInput(session, "situation", selected = "")
    updateRadioButtons(session, "hl", selected = "mid")
  })
  
  output$syn_hint <- renderUI({
    q_raw <- input$q %||% ""
    q_raw <- str_trim(as.character(q_raw))
    if (!nzchar(q_raw)) return(NULL)
    
    terms <- expand_with_concepts(q_raw, syn, themes)
    extra <- setdiff(unique(terms), tolower(q_raw))
    if (length(extra) == 0) return(NULL)
    
    div(class = "meta", paste0("Andere woorden die we ook meenemen: ", paste(extra[1:min(6, length(extra))], collapse = ", ")))
  })
  
  filtered <- reactive({
    df <- content_ui
    if (!is.null(input$cat) && input$cat != "Alles") df <- df %>% filter(category == input$cat)
    
    q_raw <- input$q %||% ""
    q_raw <- str_trim(as.character(q_raw))
    if (!nzchar(q_raw)) return(df)
    
    q_terms <- expand_with_concepts(q_raw, syn, themes)
    
    low_title <- tolower(df$title %||% "")
    low_short <- tolower(df$short %||% "")
    low_alias <- tolower(df$aliases %||% "")
    
    hit <- rep(FALSE, nrow(df))
    for (t in q_terms) {
      hit <- hit | grepl(t, low_title, fixed = TRUE) | grepl(t, low_short, fixed = TRUE)
      if (isTRUE(input$use_aliases)) hit <- hit | grepl(t, low_alias, fixed = TRUE)
    }
    
    if (!any(hit)) {
      vocab <- unique(c(content_ui$title, content_ui$aliases))
      vocab <- vocab[!is.na(vocab)]
      m <- best_spelling_match(q_raw, vocab, max_dist = 2)
      
      if (!is.na(m)) {
        hit <- grepl(m, low_title, fixed = TRUE) | grepl(m, low_short, fixed = TRUE)
        if (isTRUE(input$use_aliases)) hit <- hit | grepl(m, low_alias, fixed = TRUE)
      }
    }
    
    df[hit, , drop = FALSE]
  })
  
  output$spell_hint <- renderUI({
    q_raw <- input$q %||% ""
    q_raw <- str_trim(as.character(q_raw))
    if (!nzchar(q_raw)) return(NULL)
    
    vocab <- unique(c(content_ui$title, content_ui$aliases))
    vocab <- vocab[!is.na(vocab)]
    m <- best_spelling_match(q_raw, vocab, max_dist = 2)
    
    if (is.na(m) || tolower(m) == tolower(q_raw)) return(NULL)
    div(class = "meta", paste0("Bedoelde je misschien: ", m, " ?"))
  })
  
  output$results_count <- renderUI({
    df <- filtered()
    div(class = "meta", paste("Aantal resultaten:", nrow(df)))
  })
  
  output$results_list <- renderUI({
    df <- filtered()
    if (nrow(df) == 0) {
      return(tagList(
        p("Geen resultaten."),
        div(class = "meta", "Tip: klik op ‘Veelgezocht’, kies ‘Zoek op situatie’, gebruik ‘Onderwerpen’, of probeer AI-navigatie.")
      ))
    }
    
    df <- df %>% slice_head(n = 50)
    groups <- split(df, df$category)
    
    tagList(lapply(names(groups), function(catname) {
      g <- groups[[catname]]
      tagList(
        tags$h4(catname),
        tagList(lapply(seq_len(nrow(g)), function(i) {
          a <- g[i, ]
          card(
            class = "result-card",
            tags$strong(a$title),
            if (nzchar(a$short)) p(a$short),
            actionButton(paste0("open_", a$doc_id), "Open", class = "btn-sm btn-primary")
          )
        })),
        tags$hr()
      )
    }))
  })
  
  observe({
    lapply(content_ui$doc_id, function(id) {
      observeEvent(input[[paste0("open_", id)]], {
        selected_id(id)
        updateNavbarPage(session, id = "topnav", selected = "detail")
      }, ignoreInit = TRUE)
    })
  })
  
  selected_row <- reactive({
    req(selected_id())
    content_ui %>% filter(doc_id == selected_id()) %>% slice(1)
  })
  
  output$detail_title <- renderText({ selected_row()$title })
  output$detail_meta <- renderText({
    a <- selected_row()
    paste(a$category, "•", a$url)
  })
  output$detail_short <- renderText({ selected_row()$short })
  output$detail_long <- renderText({ selected_row()$long })
  output$detail_next <- renderText({
    x <- selected_row()$next_step
    if (is.na(x) || !nzchar(x)) "—" else x
  })
  output$detail_disclaimer <- renderText({
    x <- selected_row()$disclaimer
    if (is.na(x) || !nzchar(x)) "—" else x
  })
  
  output$detail_layer <- renderUI({
    a <- selected_row()
    blocks <- hl_blocks %>% filter(doc_id == a$doc_id)
    
    if (input$hl == "low") {
      bsum <- blocks %>% filter(hl_level == "low", type == "summary") %>% slice(1)
      steps <- blocks %>% filter(hl_level == "low", type == "step") %>% slice_head(n = 3)
      g <- extract_glossary_hits(paste(a$short, a$long), glossary, max_n = 6)
      
      card(
        div(class = "teaser-kicker", "Basis – kern & stappen"),
        if (nrow(bsum) > 0) p(bsum$content[1]),
        if (nrow(steps) > 0) tags$ul(lapply(steps$content, tags$li)),
        if (nrow(g) > 0) tagList(
          tags$hr(),
          div(class = "teaser-kicker", "Woordenlijst"),
          tags$ul(lapply(seq_len(nrow(g)), function(i) {
            tags$li(tags$strong(g$term[i]), ": ", g$uitleg[i])
          }))
        )
      )
    } else if (input$hl == "mid") {
      bsum <- blocks %>% filter(hl_level == "mid", type == "summary") %>% slice(1)
      tips <- blocks %>% filter(hl_level == "mid", type == "tip") %>% slice_head(n = 3)
      
      card(
        div(class = "teaser-kicker", "Standaard – samenvatting & tips"),
        if (nrow(bsum) > 0) p(bsum$content[1]),
        if (nrow(tips) > 0) tags$ul(lapply(tips$content, tags$li))
      )
    } else {
      NULL
    }
  })
  
  observeEvent(input$back_to_search, {
    updateNavbarPage(session, id = "topnav", selected = "search")
  })
}

shinyApp(ui, server)

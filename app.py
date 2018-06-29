import dash

app = dash.Dash()
server = app.server
app.config.suppress_callback_exceptions = True


# Append Bootstrap

bootstrap_css = "https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css"

dashboard_css = "https://getbootstrap.com/docs/4.1/examples/dashboard/dashboard.css"

app.css.append_css({

    "external_url": bootstrap_css
})

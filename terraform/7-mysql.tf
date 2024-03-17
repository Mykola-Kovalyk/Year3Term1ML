

resource "google_sql_database_instance" "default" {
  name             = "mysql-instance"
  database_version = "MYSQL_5_7"

  settings {
    tier = "db-f1-micro"
    ip_configuration {
      authorized_networks {
        name  = "All"
        value = "0.0.0.0/0"
      }
    }
  }

  deletion_protection = false

  depends_on = [google_project_service.cloud_sql_admin_api]
}

resource "google_sql_database" "default" {
  name     = "mlflow"
  instance = google_sql_database_instance.default.name
}

resource "google_sql_user" "default" {
  name     = var.mysql_user
  instance = google_sql_database_instance.default.name
  password = var.mysql_password
}

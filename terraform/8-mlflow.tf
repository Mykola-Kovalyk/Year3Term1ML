


resource "google_cloud_run_v2_service" "mlflow" {
  name     = "mlflow"
  location = local.region

  template {
    containers {
      image = "${local.artifact_registry_location}/${local.project_id}/${local.artifact_registry_name}/mlflow:v2.12.2"
      args = [
        "--backend-store-uri",
        "mysql://${var.mysql_user}:${var.mysql_password}@${google_sql_database_instance.default.public_ip_address}/${google_sql_database.default.name}",
        "--artifacts-destination",
        google_storage_bucket.mlflow_artifact_state.url
      ]
      ports {
        container_port = 5000
      }
      resources {
        limits = {
          cpu    = "2"
          memory = "1024Mi"
        }
      }
    }
    service_account = google_service_account.mlflow_account.email
  }


  depends_on = [google_project_service.cloud_run]
}


resource "google_storage_bucket" "mlflow_artifact_state" {
  name     = "mlflow_artifact_state"
  location = local.region

  uniform_bucket_level_access = true
}

resource "google_cloud_run_service_iam_binding" "allow_public_access" {
  location = google_cloud_run_v2_service.mlflow.location
  service  = google_cloud_run_v2_service.mlflow.name
  role     = "roles/run.invoker"
  members = [
    "allUsers"
  ]
}

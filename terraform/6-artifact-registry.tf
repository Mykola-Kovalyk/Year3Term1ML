resource "google_artifact_registry_repository" "my_repository" {
  repository_id = local.artifact_registry_name
  format        = "DOCKER"

  labels = {
    environment = "dev"
  }

  depends_on  = [google_project_service.artifact_registry]
  description = "Artifact Registry Repository"
}

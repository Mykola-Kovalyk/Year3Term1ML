resource "google_service_account" "deployed_instance_account" {
  account_id   = "deployed-instance"
  display_name = "Account used by model when running"
}

resource "google_service_account" "github_actions_account" {
  account_id   = "github-actions"
  display_name = "Account used by Github Actions to deploy to GKE"
}

resource "google_project_iam_binding" "github_actions_editor_binding" {
  project = local.project_id
  role    = "roles/editor"

  members = [
    "serviceAccount:${google_service_account.github_actions_account.email}"
  ]
}

resource "google_service_account" "kubernetes_node" {
  account_id   = "kubernetes-node"
  display_name = "Account used by GKE nodes"
}

resource "google_project_iam_binding" "kubernetes_node_cluster_binding" {
  project = local.project_id
  role    = "roles/container.clusterViewer"

  members = [
    "serviceAccount:${google_service_account.kubernetes_node.email}"
  ]
}

resource "google_project_iam_binding" "kubernetes_node_artifact_registry_binding" {
  project = local.project_id
  role    = "roles/artifactregistry.reader"

  members = [
    "serviceAccount:${google_service_account.kubernetes_node.email}"
  ]
}

resource "google_service_account" "mlflow_account" {
  account_id   = "mlflow"
  display_name = "Account used by MLflow to access GCS"
}

resource "google_project_iam_binding" "mlflow_binding" {
  project = local.project_id
  role    = "roles/storage.objectUser"

  members = [
    "serviceAccount:${google_service_account.mlflow_account.email}"
  ]
}

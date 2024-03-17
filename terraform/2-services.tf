resource "google_project_service" "compute" {
  project                    = local.project_id
  service                    = "compute.googleapis.com"
  disable_dependent_services = true
}

resource "google_project_service" "container" {
  project                    = local.project_id
  service                    = "container.googleapis.com"
  disable_dependent_services = true
}


resource "google_project_service" "secretmanager" {
  project                    = local.project_id
  service                    = "secretmanager.googleapis.com"
  disable_dependent_services = true
}


resource "google_project_service" "servicenetworking" {
  project                    = local.project_id
  service                    = "servicenetworking.googleapis.com"
  disable_dependent_services = true
}

resource "google_project_service" "serviceusage" {
  project                    = local.project_id
  service                    = "serviceusage.googleapis.com"
  disable_dependent_services = true
}

resource "google_project_service" "gcp_resource_manager_api" {
  project                    = local.project_id
  service                    = "cloudresourcemanager.googleapis.com"
  disable_dependent_services = true
}

resource "google_project_service" "cloud_sql_admin_api" {
  project                    = local.project_id
  service                    = "sqladmin.googleapis.com"
  disable_dependent_services = true
}

resource "google_project_service" "cloud_run" {
  project                    = local.project_id
  service                    = "run.googleapis.com"
  disable_dependent_services = true
}

resource "google_project_service" "artifact_registry" {
  project                    = local.project_id
  service                    = "artifactregistry.googleapis.com"
  disable_dependent_services = true
}

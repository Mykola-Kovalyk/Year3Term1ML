resource "google_container_cluster" "primary" {
  name               = "primary"
  location           = local.zone
  initial_node_count = 3

  addons_config {
    http_load_balancing {
      disabled = true
    }
    horizontal_pod_autoscaling {
      disabled = true
    }
  }

  node_config {
    service_account = google_service_account.kubernetes_node.email
  }

  depends_on = [google_project_service.compute, google_project_service.container]
}


data "google_client_config" "current" {}

data "kubectl_filename_list" "manifests" {
    pattern = "../k8s/*.yaml"
}

resource "kubectl_manifest" "apply_manifests" {
    count = length(data.kubectl_filename_list.manifests.matches)
    yaml_body = file(element(data.kubectl_filename_list.manifests.matches, count.index))
}


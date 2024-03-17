

data "google_compute_network" "default" {
  name = "default"

  depends_on = [google_project_service.compute, google_project_service.container]
}

resource "google_compute_firewall" "default_http" {
  name    = "ml-default-http-firewall"
  network = data.google_compute_network.default.name

  priority                = 500
  source_ranges           = ["0.0.0.0/0"]
  target_service_accounts = [google_service_account.deployed_instance_account.email, google_service_account.kubernetes_node.email]

  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }

  depends_on = [google_project_service.compute, google_project_service.container]

}


resource "google_compute_firewall" "mysql" {
  name    = "ml-mysql-firewall"
  network = data.google_compute_network.default.name

  priority                = 500
  source_service_accounts = [google_service_account.deployed_instance_account.email, google_service_account.kubernetes_node.email]

  allow {
    protocol = "tcp"
    ports    = ["3306"]
  }

  depends_on = [google_project_service.compute, google_project_service.container]
}


resource "google_compute_firewall" "block_rest" {
  name    = "ml-block-rest-firewall"
  network = data.google_compute_network.default.name

  priority      = 5000
  source_ranges = ["0.0.0.0/0"]

  deny {
    protocol = "tcp"
    ports    = ["1-65535"]
  }

  depends_on = [google_project_service.compute, google_project_service.container]
}

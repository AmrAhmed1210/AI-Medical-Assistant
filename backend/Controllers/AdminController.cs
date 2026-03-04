using Microsoft.AspNetCore.Mvc;
using backend.Models;

[ApiController]
[Route("api/[controller]")]
public class AdminController : ControllerBase {
    [HttpGet("dashboard-stats")]
    public IActionResult GetDashboardStats() {
        // هذه الأرقام ستأتي لاحقاً من قاعدة البيانات
        var stats = new AdminStats {
            TotalUsers = 12480,
            TotalDoctors = 342,
            TotalPatients = 11850,
            Revenue = 284500,
            ActiveConsultations = 89
        };
        return Ok(stats);
    }
}

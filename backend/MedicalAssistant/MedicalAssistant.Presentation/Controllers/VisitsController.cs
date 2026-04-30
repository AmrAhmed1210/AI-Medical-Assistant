using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.PatientVisits;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Linq;
using System.Threading.Tasks;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("api/visits")]
    [Authorize]
    public class VisitsController : ControllerBase
    {
        private readonly IPatientVisitService _visitService;
        private readonly IUnitOfWork _unitOfWork;

        public VisitsController(IPatientVisitService visitService, IUnitOfWork unitOfWork)
        {
            _visitService = visitService;
            _unitOfWork = unitOfWork;
        }

        private int? GetUserIdFromToken()
        {
            var uid = User.FindFirst("UserId")?.Value ?? User.FindFirst(System.Security.Claims.ClaimTypes.NameIdentifier)?.Value;
            return int.TryParse(uid, out var id) ? id : null;
        }

        private int? GetPatientIdFromToken()
        {
            var pid = User.FindFirst("PatientId")?.Value;
            return int.TryParse(pid, out var id) ? id : null;
        }

        // POST /api/visits  (Doctor)
        [HttpPost]
        [Authorize(Roles = "Doctor")]
        public async Task<IActionResult> Open([FromBody] CreateVisitDto dto)
        {
            var userId = GetUserIdFromToken();
            if (!userId.HasValue) return Unauthorized(new { message = "Invalid token." });

            var created = await _visitService.OpenVisitAsync(userId.Value, dto);
            return Ok(created);
        }

        // GET /api/visits/{id}  (Doctor, Patient(own))
        [HttpGet("{id:int}")]
        [Authorize(Roles = "Doctor,Patient")]
        public async Task<IActionResult> GetById(int id)
        {
            var visit = await _visitService.GetVisitAsync(id);
            if (visit == null) return NotFound(new { message = "Visit not found." });

            if (User.IsInRole("Patient"))
            {
                var pid = GetPatientIdFromToken();
                if (!pid.HasValue || pid.Value != visit.PatientId) return Forbid();
            }

            return Ok(visit);
        }

        // PATCH /api/visits/{id}  (Doctor)
        [HttpPatch("{id:int}")]
        [Authorize(Roles = "Doctor")]
        public async Task<IActionResult> Update(int id, [FromBody] UpdateVisitDto dto)
        {
            var userId = GetUserIdFromToken();
            if (!userId.HasValue) return Unauthorized(new { message = "Invalid token." });

            var updated = await _visitService.UpdateVisitAsync(userId.Value, id, dto);
            if (updated == null) return NotFound(new { message = "Visit not found." });
            return Ok(updated);
        }

        // PATCH /api/visits/{id}/close  (Doctor)
        [HttpPatch("{id:int}/close")]
        [Authorize(Roles = "Doctor")]
        public async Task<IActionResult> Close(int id)
        {
            var userId = GetUserIdFromToken();
            if (!userId.HasValue) return Unauthorized(new { message = "Invalid token." });

            var ok = await _visitService.CloseVisitAsync(userId.Value, id);
            if (!ok) return NotFound(new { message = "Visit not found." });
            return Ok(new { message = "Visit closed." });
        }

        // GET /api/patients/{id}/visits  (Doctor, Patient(own))
        [HttpGet("/api/patients/{id:int}/visits")]
        [Authorize(Roles = "Doctor,Patient")]
        public async Task<IActionResult> GetForPatient(int id)
        {
            if (User.IsInRole("Patient"))
            {
                var pid = GetPatientIdFromToken();
                if (!pid.HasValue || pid.Value != id) return Forbid();
            }

            var visits = await _visitService.GetVisitsForPatientAsync(id);
            return Ok(visits);
        }

        // GET /api/visits/doctor/today  (Doctor)
        [HttpGet("doctor/today")]
        [Authorize(Roles = "Doctor")]
        public async Task<IActionResult> GetDoctorToday()
        {
            var userId = GetUserIdFromToken();
            if (!userId.HasValue) return Unauthorized(new { message = "Invalid token." });

            var visits = await _visitService.GetTodayVisitsForDoctorAsync(userId.Value);
            return Ok(visits);
        }

        // GET /api/visits/{id}/summary  (Doctor)
        [HttpGet("{id:int}/summary")]
        [Authorize(Roles = "Doctor")]
        public async Task<IActionResult> GetSummary(int id)
        {
            var userId = GetUserIdFromToken();
            if (!userId.HasValue) return Unauthorized(new { message = "Invalid token." });

            var summary = await _visitService.GetVisitSummaryAsync(userId.Value, id);
            if (summary == null) return NotFound(new { message = "Visit not found." });

            return Ok(summary);
        }

        // GET /api/visits/{id}/summary-pdf  (Doctor)
        // Placeholder: returns JSON for now.
        [HttpGet("{id:int}/summary-pdf")]
        [Authorize(Roles = "Doctor")]
        public async Task<IActionResult> GetSummaryPdf(int id)
        {
            var userId = GetUserIdFromToken();
            if (!userId.HasValue) return Unauthorized(new { message = "Invalid token." });

            var summary = await _visitService.GetVisitSummaryAsync(userId.Value, id);
            if (summary == null) return NotFound(new { message = "Visit not found." });

            return Ok(new { message = "PDF generation not implemented", summary });
        }
    }
}

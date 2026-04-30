using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.PatientRecords;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System;
using System.Linq;
using System.Threading.Tasks;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("api")]
    [Authorize]
    public class VitalsController : ControllerBase
    {
        private readonly IPatientRecordService _patientRecordService;

        public VitalsController(IPatientRecordService patientRecordService)
        {
            _patientRecordService = patientRecordService;
        }

        private int? GetPatientIdFromToken()
        {
            var pid = User.FindFirst("PatientId")?.Value;
            return int.TryParse(pid, out var id) ? id : null;
        }

        private bool IsDoctor() => User.IsInRole("Doctor");
        private bool IsNurse() => User.IsInRole("Nurse");

        private bool IsOwnPatient(int patientId)
        {
            var tokenPid = GetPatientIdFromToken();
            return tokenPid.HasValue && tokenPid.Value == patientId;
        }

        // GET /api/patients/{id}/vitals?type=&from=&to=  (Doctor, Patient(own))
        [HttpGet("patients/{id:int}/vitals")]
        [Authorize(Roles = "Doctor,Patient")]
        public async Task<IActionResult> GetForPatient(int id, [FromQuery] string? type = null, [FromQuery] DateTime? from = null, [FromQuery] DateTime? to = null)
        {
            if (!IsDoctor() && !IsOwnPatient(id)) return Forbid();

            var items = await _patientRecordService.GetVitalsAsync(id, type, from, to);
            var dto = items.Select(v => new VitalReadingDto(
                v.Id,
                v.PatientId,
                v.ChronicDiseaseMonitorId == 0 ? null : v.ChronicDiseaseMonitorId,
                v.ReadingType,
                v.Value,
                v.Value2,
                v.Unit,
                v.SugarReadingContext,
                v.IsNormal,
                v.RecordedBy,
                v.Notes,
                v.RecordedAt
            ));

            return Ok(dto);
        }

        // POST /api/patients/{id}/vitals  (Doctor,Nurse,Patient)
        [HttpPost("patients/{id:int}/vitals")]
        [Authorize(Roles = "Doctor,Nurse,Patient")]
        public async Task<IActionResult> Create(int id, [FromBody] CreateVitalReadingDto dto)
        {
            if (User.IsInRole("Patient") && !IsOwnPatient(id)) return Forbid();

            var recordedBy = dto.RecordedBy;
            if (string.IsNullOrWhiteSpace(recordedBy))
            {
                if (IsDoctor()) recordedBy = "Doctor";
                else if (IsNurse()) recordedBy = "Nurse";
                else recordedBy = "Patient";
            }

            var entity = new VitalReading
            {
                ChronicDiseaseMonitorId = dto.ChronicDiseaseMonitorId ?? 0,
                ReadingType = dto.ReadingType,
                Value = dto.Value,
                Value2 = dto.Value2,
                Unit = dto.Unit,
                SugarReadingContext = dto.SugarReadingContext,
                IsNormal = dto.IsNormal,
                Notes = dto.Notes,
                RecordedBy = recordedBy,
                RecordedAt = DateTime.UtcNow,
            };

            var created = await _patientRecordService.AddVitalAsync(id, entity);

            return Ok(new VitalReadingDto(
                created.Id,
                created.PatientId,
                created.ChronicDiseaseMonitorId == 0 ? null : created.ChronicDiseaseMonitorId,
                created.ReadingType,
                created.Value,
                created.Value2,
                created.Unit,
                created.SugarReadingContext,
                created.IsNormal,
                created.RecordedBy,
                created.Notes,
                created.RecordedAt
            ));
        }

        // GET /api/patients/{id}/vitals/latest?type=  (Doctor, Patient(own))
        [HttpGet("patients/{id:int}/vitals/latest")]
        [Authorize(Roles = "Doctor,Patient")]
        public async Task<IActionResult> GetLatest(int id, [FromQuery] string type)
        {
            if (!IsDoctor() && !IsOwnPatient(id)) return Forbid();
            if (string.IsNullOrWhiteSpace(type)) return BadRequest(new { message = "type is required" });

            var item = await _patientRecordService.GetLatestVitalAsync(id, type);
            if (item == null) return NotFound(new { message = "No reading found." });

            return Ok(new VitalReadingDto(
                item.Id,
                item.PatientId,
                item.ChronicDiseaseMonitorId == 0 ? null : item.ChronicDiseaseMonitorId,
                item.ReadingType,
                item.Value,
                item.Value2,
                item.Unit,
                item.SugarReadingContext,
                item.IsNormal,
                item.RecordedBy,
                item.Notes,
                item.RecordedAt
            ));
        }

        // GET /api/patients/{id}/vitals/trend?type=&days=30  (Doctor, Patient(own))
        [HttpGet("patients/{id:int}/vitals/trend")]
        [Authorize(Roles = "Doctor,Patient")]
        public async Task<IActionResult> GetTrend(int id, [FromQuery] string type, [FromQuery] int days = 30)
        {
            if (!IsDoctor() && !IsOwnPatient(id)) return Forbid();
            if (string.IsNullOrWhiteSpace(type)) return BadRequest(new { message = "type is required" });

            var data = await _patientRecordService.GetVitalTrendAsync(id, type, days);
            return Ok(data);
        }

        // DELETE /api/vitals/{id}  (Doctor,Admin)
        [HttpDelete("vitals/{id:int}")]
        [Authorize(Roles = "Doctor,Admin")]
        public async Task<IActionResult> Delete(int id)
        {
            var ok = await _patientRecordService.DeleteVitalAsync(id);
            if (!ok) return NotFound(new { message = "Vital reading not found." });
            return Ok(new { message = "Vital reading deleted." });
        }
    }
}

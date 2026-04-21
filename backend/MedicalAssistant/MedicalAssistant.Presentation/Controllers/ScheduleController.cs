using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.DoctorDTOs;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace MedicalAssistant.Presentation.Controllers;

[ApiController]
[Route("api/schedule")]
public class ScheduleController : ControllerBase
{
    private readonly IDoctorService _doctorService;
    private readonly INotificationService _notificationService;

    public ScheduleController(IDoctorService doctorService, INotificationService notificationService)
    {
        _doctorService = doctorService;
        _notificationService = notificationService;
    }

    private int GetDoctorUserId()
    {
        var userIdClaim = User.Claims.FirstOrDefault(c => c.Type == "UserId")?.Value;
        return int.TryParse(userIdClaim, out var id) ? id : 0;
    }

    [AllowAnonymous]
    [HttpGet("{doctorId:int}")]
    public async Task<ActionResult<DoctorScheduleDto>> GetByDoctorId(int doctorId)
    {
        var result = await _doctorService.GetScheduleAsync(doctorId);
        if (result == null) return NotFound(new { message = "Schedule not found." });
        return Ok(result);
    }

    [Authorize(Roles = "Doctor")]
    [HttpGet("me")]
    public async Task<ActionResult<DoctorScheduleDto>> GetMySchedule()
    {
        var doctorUserId = GetDoctorUserId();
        if (doctorUserId <= 0) return Unauthorized();

        var result = await _doctorService.GetMyScheduleAsync(doctorUserId);
        if (result == null) return NotFound(new { message = "Schedule not found." });
        return Ok(result);
    }

    [Authorize(Roles = "Doctor")]
    [HttpPut("me")]
    public async Task<IActionResult> UpdateMySchedule([FromBody] UpdateDoctorScheduleRequest request)
    {
        var doctorUserId = GetDoctorUserId();
        if (doctorUserId <= 0) return Unauthorized();

        await _doctorService.UpdateScheduleAsync(doctorUserId, request);
        var schedule = await _doctorService.GetMyScheduleAsync(doctorUserId);
        if (schedule == null) return NotFound(new { message = "Schedule not found." });

        await _notificationService.NotifyScheduleUpdated(
            schedule.DoctorId,
            schedule.DoctorName,
            schedule.IsMobileEnabled);

        return Ok(schedule);
    }
}

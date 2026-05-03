using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.Secretary;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Security.Claims;

namespace MedicalAssistant.Presentation.Controllers;

[ApiController]
[Route("api/[controller]")]
[Authorize]
public class SecretaryController : ControllerBase
{
    private readonly ISecretaryService _secretaryService;

    public SecretaryController(ISecretaryService secretaryService)
    {
        _secretaryService = secretaryService;
    }

    private int GetCurrentUserId()
    {
        var userIdStr = User.FindFirst(ClaimTypes.NameIdentifier)?.Value 
                     ?? User.FindFirst("UserId")?.Value;
        return int.Parse(userIdStr ?? "0");
    }

    [HttpPost("add")]
    [Authorize(Roles = "Doctor,Admin")]
    public async Task<IActionResult> AddSecretary([FromBody] CreateSecretaryDto dto)
    {
        try
        {
            var result = await _secretaryService.AddSecretaryAsync(GetCurrentUserId(), dto);
            return Ok(result);
        }
        catch (Exception ex)
        {
            return BadRequest(new { message = ex.Message });
        }
    }

    [HttpGet("my-secretaries")]
    [Authorize(Roles = "Doctor,Admin")]
    public async Task<IActionResult> GetMySecretaries()
    {
        var result = await _secretaryService.GetSecretariesForDoctorAsync(GetCurrentUserId());
        return Ok(result);
    }

    [HttpDelete("{id}")]
    [Authorize(Roles = "Doctor,Admin")]
    public async Task<IActionResult> DeleteSecretary(int id)
    {
        var success = await _secretaryService.DeleteSecretaryAsync(GetCurrentUserId(), id);
        return success ? Ok() : NotFound();
    }
}
